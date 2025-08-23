#!/usr/bin/env python3
"""
Myanmar News Scraper for MyAwady News Website
Focused scraper for MyAwady military news category
"""

import requests
from bs4 import BeautifulSoup
import json
import os
import time
import re
from datetime import datetime
from urllib.parse import urljoin
import logging
from typing import List, Dict, Optional
import random
import urllib3

# Disable SSL warnings for MyAwady site
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MyAwadyScraper:
    def __init__(self, output_dir: str = None, session_name: str | None = None):
        # Use the correct data directory from utils
        if output_dir is None:
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
            from utils import get_data_directories
            dirs = get_data_directories()
            output_dir = dirs['raw_scraped']
        # Create session folder
        if session_name is None:
            session_name = datetime.now().strftime("session_%Y%m%d_%H%M%S")
        
        self.session_dir = os.path.join(output_dir, session_name)
        self.output_dir = self.session_dir
        os.makedirs(self.session_dir, exist_ok=True)
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        # Disable SSL verification for MyAwady if needed
        self.session.verify = False
        
        # Myanmar Unicode pattern for text filtering
        self.myanmar_pattern = re.compile(r'[\u1000-\u109F\u1040-\u1049\uAA60-\uAA7F]+')
        
        # Load previously scraped URLs to avoid duplicates
        self.scraped_urls_file = os.path.join(output_dir, "scraped_urls.json")
        self.visited_urls = self.load_scraped_urls()
        
        logger.info(f"Session directory: {self.session_dir}")
        logger.info(f"Previously scraped URLs: {len(self.visited_urls)}")
        
    def is_myanmar_text(self, text: str) -> bool:
        """Check if text contains Myanmar script characters."""
        if not text.strip() or len(text) < 5:
            return False
            
        # Count Myanmar characters more accurately
        myanmar_matches = self.myanmar_pattern.findall(text)
        myanmar_char_count = sum(len(match) for match in myanmar_matches)
        
        # Count all non-whitespace characters
        total_chars = len([c for c in text if not c.isspace()])
        
        if total_chars == 0:
            return False
            
        myanmar_ratio = myanmar_char_count / total_chars
        
        # Lower threshold for better detection and require at least some Myanmar chars
        return myanmar_char_count >= 3 and myanmar_ratio > 0.2
    
    def clean_text(self, text: str) -> str:
        """Clean Myanmar text and remove invisible Unicode characters."""
        # Remove invisible Unicode characters first
        text = self.remove_invisible_chars(text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        # Remove HTML entities
        text = re.sub(r'&[a-zA-Z0-9#]+;', '', text)
        # Keep Myanmar script, numbers, and basic punctuation
        text = re.sub(r'[^\u1000-\u109F\u1040-\u1049\uAA60-\uAA7F\s\w.,!?()-]', '', text)
        return text.strip()
    
    def remove_invisible_chars(self, text: str) -> str:
        """Remove invisible/problematic Unicode characters from text."""
        # Characters to remove (invisible/problematic ones)
        invisible_chars = [
            '\u200B',  # Zero Width Space
            '\u200C',  # Zero Width Non-Joiner  
            '\u200D',  # Zero Width Joiner
            '\u2060',  # Word Joiner
            '\uFEFF',  # Byte Order Mark
            '\u00A0',  # Non-breaking Space (the main culprit from MyAwady)
            '\u2009',  # Thin Space
            '\u200A',  # Hair Space
            '\u2028',  # Line Separator
            '\u2029',  # Paragraph Separator
        ]
        
        cleaned = text
        for char in invisible_chars:
            cleaned = cleaned.replace(char, ' ')  # Replace with regular space
        
        # Replace multiple spaces with single spaces
        cleaned = re.sub(r' +', ' ', cleaned)
        
        return cleaned.strip()
    
    def get_page(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch and parse webpage."""        
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
        except Exception as e:
            logger.warning(f"Error fetching {url}: {e}")
            return None
    
    def extract_article_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract article links from MyAwady news page."""
        links = []
        
        # Find all links that look like articles with /node/ pattern
        all_links = soup.find_all('a', href=True)
        
        for link in all_links:
            href = link.get('href')  # type: ignore
            if not href or not isinstance(href, str):
                continue
                
            # Handle both relative and absolute URLs for MyAwady /node/ pattern
            if href.startswith('/node/'):
                full_url = urljoin(base_url, href)
            elif '/node/' in href and 'myawady.net.mm' in href:
                full_url = href
            else:
                continue
                
            # Add unique links (don't filter visited_urls here - done in pagination)
            if full_url not in links:
                links.append(full_url)
        
        return links
    
    def scrape_article(self, url: str) -> Optional[Dict]:
        """Scrape a single MyAwady article - extract only the actual article content."""
        soup = self.get_page(url)
        if not soup:
            return None
        
        # Remove unwanted elements that commonly contain navigation/menu content
        unwanted_selectors = [
            'script', 'style', 'nav', 'header', 'footer', 'menu', 'aside', 
            'form', '.navigation', '.nav', '.menu', '.sidebar', '.widget',
            '.breadcrumbs', '.tags', '.categories', '.related', '.share',
            '.comments', '.social', '.author', '.metadata', '.byline'
        ]
        
        for selector in unwanted_selectors:
            for element in soup.select(selector):
                element.decompose()
        
        # Remove specific MyAwady taxonomy elements
        taxonomy_selectors = [
            '.field-name-field-new-taxs',  # The taxonomy field container
            '.field-type-taxonomy-term-reference',
            'a[href^="/taxonomy/term"]'  # Links to taxonomy terms
        ]
        
        for selector in taxonomy_selectors:
            for element in soup.select(selector):
                element.decompose()
        
        # Extract title - try different selectors for MyAwady
        raw_title = ""
        title_selectors = ['h1', '.post-title', '.entry-title', '.article-title', '.node-title', 'title']
        
        for selector in title_selectors:
            title_elem = soup.select_one(selector)
            if title_elem:
                title_text = title_elem.get_text().strip()
                # Simple check - just needs to be Myanmar text and reasonable length
                if title_text and self.is_myanmar_text(title_text) and len(title_text) > 5:
                    raw_title = title_text
                    break
        
        # Extract content using MyAwady-specific strategy
        raw_content = ""
        
        # Strategy 1: Try to find the main article body container
        article_body = soup.select_one('.field-name-body .field-items')
        if article_body:
            # Get all paragraphs from the body field
            paragraphs = article_body.find_all('p')
            article_paragraphs = []
            
            for p in paragraphs:
                p_text = p.get_text().strip()
                if (p_text and self.is_myanmar_text(p_text) 
                    and len(p_text) > 15
                    and not self.is_navigation_text(p_text)):
                    article_paragraphs.append(p_text)
            
            if article_paragraphs:
                raw_content = ' '.join(article_paragraphs)
                logger.debug(f"Strategy 1: Found {len(article_paragraphs)} paragraphs from article body")
        
        # Fallback: Use broader search if no content found
        if not raw_content:
            # Get all paragraphs and divs that might contain content
            all_paragraphs = soup.find_all(['p', 'div'])
            article_paragraphs = []
            
            for p in all_paragraphs:
                p_text = p.get_text().strip()
                # Filter for good content
                if (p_text and self.is_myanmar_text(p_text) 
                    and len(p_text) > 20  # Reasonable length
                    and not self.is_navigation_text(p_text)):
                    article_paragraphs.append(p_text)
            
            # Filter out the very first paragraph if it looks like navigation
            if article_paragraphs and len(article_paragraphs[0]) > 300:
                # First paragraph is very long, might be navigation menu
                if self.is_navigation_text(article_paragraphs[0]):
                    article_paragraphs = article_paragraphs[1:]
            
            if article_paragraphs:
                raw_content = ' '.join(article_paragraphs)
                logger.debug(f"Fallback: Found {len(article_paragraphs)} paragraphs")
        
        # Must have both title and content
        if not raw_title or not raw_content or len(raw_content) < 50:
            return None
        
        # Clean the title and content to remove invisible Unicode characters
        clean_title = self.clean_text(raw_title)
        clean_content = self.clean_text(raw_content)
        
        # Final validation after cleaning
        if not clean_title or not clean_content or len(clean_content) < 30:
            return None
        
        # Mark URL as successfully scraped
        self.visited_urls.add(url)
        
        return {
            'url': url,
            'title': clean_title,
            'content': clean_content,
            'scraped_at': datetime.now().isoformat()
        }
    
    def is_navigation_text(self, text: str) -> bool:
        """Check if text looks like navigation/menu content - enhanced for MyAwady."""
        # MyAwady-specific taxonomy terms (high priority to filter out)
        taxonomy_terms = [
            'ခေါင်းကြီးပိုင်းသတင်း',
            'ပြည်တွင်းသတင်း', 
            'တပ်မတော်သတင်း',
            'စီးပွားရေးသတင်း',
            'ကျန်းမာရေးသတင်း',
            'ပညာရေးသတင်း'
        ]
        
        # If text contains multiple taxonomy terms, it's navigation
        taxonomy_count = sum(1 for term in taxonomy_terms if term in text)
        if taxonomy_count >= 2:
            return True
            
        # If text is ONLY taxonomy terms repeated, it's navigation
        if taxonomy_count >= 1 and len(text) < 200:
            # Remove all taxonomy terms and see what's left
            remaining_text = text
            for term in taxonomy_terms:
                remaining_text = remaining_text.replace(term, '')
            # If very little content remains after removing taxonomy terms
            if len(remaining_text.strip()) < 20:
                return True
        
        # Common navigation indicators for MyAwady
        nav_indicators = [
            'home', 'login', 'english', 'search', 'menu', 
            'follow us', 'contact', 'privacy', 'about',
            'myawady', 'မြဝတီ', 'ပင်မစာမျက်နှာ', 'toggle navigation'
        ]
        
        text_lower = text.lower()
        
        # Check for navigation patterns
        nav_matches = sum(1 for indicator in nav_indicators if indicator in text_lower)
        if nav_matches >= 2:  # Multiple navigation indicators
            return True
        
        # Check for long category-like text (the problematic navigation menu)
        if (len(text) > 200 and len(text.split()) > 30 
            and ('/' in text or 'သတင်း' in text[:100])):  # Lots of category separators
            return True
        
        # Check for repetitive short phrases (common in menus)
        words = text.split()
        if len(words) > 15 and len(set(words)) < len(words) * 0.5:  # Too much repetition
            return True
        
        # Check for metadata-like text (submission info, etc.)
        if 'submitted by' in text_lower or 'mwd_webportal' in text_lower:
            return True
            
        return False
    
    def load_scraped_urls(self) -> set:
        """Load previously scraped URLs from file."""
        if os.path.exists(self.scraped_urls_file):
            try:
                with open(self.scraped_urls_file, 'r', encoding='utf-8') as f:
                    urls_data = json.load(f)
                    return set(urls_data.get('scraped_urls', []))
            except Exception as e:
                logger.warning(f"Could not load scraped URLs: {e}")
        return set()
    
    def save_scraped_urls(self):
        """Save scraped URLs to file for future sessions."""
        try:
            urls_data = {
                'scraped_urls': list(self.visited_urls),
                'last_updated': datetime.now().isoformat(),
                'total_urls': len(self.visited_urls)
            }
            with open(self.scraped_urls_file, 'w', encoding='utf-8') as f:
                json.dump(urls_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Could not save scraped URLs: {e}")
    
    def scrape_myawady_with_pagination(self, base_url: str = "https://www.myawady.net.mm/militray", max_articles: int = 100, max_pages: int = 10) -> List[Dict]:
        """Scrape MyAwady articles with pagination support, skipping already scraped URLs."""
        logger.info(f"Starting MyAwady scraping from {base_url}")
        logger.info(f"Target: {max_articles} NEW articles (skipping {len(self.visited_urls)} already scraped)")
        logger.info(f"Will continue searching until {max_articles} new articles found or no more pages exist")
        
        all_articles = []
        new_article_links = []
        duplicates_found = 0
        page_num = 1
        
        # Continue searching until we find enough new articles or no more pages exist
        while len(new_article_links) < max_articles:
            if page_num == 1:
                page_url = base_url
            else:
                # MyAwady might use different pagination format - adjust as needed
                page_url = f"{base_url}?page={page_num-1}"  # Often 0-indexed
            
            logger.info(f"Fetching page {page_num}: {page_url} (need {max_articles - len(new_article_links)} more new articles)")
            soup = self.get_page(page_url)
            
            if not soup:
                logger.warning(f"Could not fetch page {page_num}")
                page_num += 1
                continue
            
            # Extract article links from this page
            page_links = self.extract_article_links(soup, base_url)
            
            if not page_links:
                logger.info(f"No more articles found on page {page_num}, website has no more pages")
                break
            
            # Filter out already scraped URLs
            new_links_on_page = []
            for link in page_links:
                if link not in self.visited_urls:
                    new_links_on_page.append(link)
                else:
                    duplicates_found += 1
            
            new_article_links.extend(new_links_on_page)
            logger.info(f"Page {page_num}: {len(new_links_on_page)} NEW links, {len(page_links) - len(new_links_on_page)} duplicates (total new so far: {len(new_article_links)})")
            
            page_num += 1
            
            # Delay between pages
            time.sleep(random.uniform(2, 3))
        
        logger.info(f"Initial collection: {len(new_article_links)} new links, {duplicates_found} duplicates skipped")
        
        # Continue scraping until we have enough Myanmar articles or run out of links
        articles_attempted = 0
        english_articles_skipped = 0
        failed_articles = 0
        
        while len(all_articles) < max_articles and articles_attempted < len(new_article_links):
            link = new_article_links[articles_attempted]
            articles_attempted += 1
            
            logger.info(f"Scraping article {articles_attempted}/{len(new_article_links)} (Myanmar articles found: {len(all_articles)}): {link}")
            
            article = self.scrape_article(link)
            if article:
                all_articles.append(article)
                content_length = len(article['content'])
                logger.info(f"✓ Myanmar article scraped: {article['title'][:60]}... ({content_length} characters)")
            else:
                # Check if this was an English article by testing the page
                soup = self.get_page(link)
                if soup:
                    article_body = soup.select_one('.field-name-body .field-items')
                    if article_body:
                        content = article_body.get_text().strip()
                        if content and len(content) > 50:
                            # Has content but not Myanmar - likely English
                            english_articles_skipped += 1
                            logger.info(f"⚪ Skipped English article: {link}")
                        else:
                            failed_articles += 1
                            logger.warning(f"✗ Failed to scrape (no content): {link}")
                    else:
                        failed_articles += 1
                        logger.warning(f"✗ Failed to scrape (no article body): {link}")
                else:
                    failed_articles += 1
                    logger.warning(f"✗ Failed to fetch page: {link}")
            
            # If we need more articles and we're running low on links, get more pages
            if (len(all_articles) < max_articles and 
                articles_attempted >= len(new_article_links) - 5):  # Close to end of current links
                
                logger.info(f"Need more Myanmar articles ({len(all_articles)}/{max_articles}), searching more pages...")
                additional_pages = 5
                
                for extra_page in range(page_num, page_num + additional_pages):
                    page_url = f"{base_url}?page={extra_page-1}"
                    logger.info(f"Fetching additional page {extra_page}: {page_url}")
                    
                    soup = self.get_page(page_url)
                    if not soup:
                        continue
                        
                    page_links = self.extract_article_links(soup, base_url)
                    if not page_links:
                        logger.info(f"No more articles on page {extra_page}, stopping additional search")
                        break
                    
                    # Filter for new links
                    for link in page_links:
                        if link not in self.visited_urls and link not in new_article_links:
                            new_article_links.append(link)
                    
                    time.sleep(random.uniform(2, 3))
                
                page_num += additional_pages
                logger.info(f"Added more links, now have {len(new_article_links)} total links to check")
            
            # Respectful delay between articles
            time.sleep(random.uniform(2, 4))
        
        logger.info(f"MyAwady scraping completed:")
        logger.info(f"  - Myanmar articles collected: {len(all_articles)}")
        logger.info(f"  - English articles skipped: {english_articles_skipped}")
        logger.info(f"  - Failed to scrape: {failed_articles}")
        logger.info(f"  - Total articles attempted: {articles_attempted}")
        
        return all_articles
    
    
    def save_raw_data(self, articles: List[Dict], filename_prefix: str = "myawady_raw_myanmar"):
        """Save raw Myanmar title + content data."""
        if not articles:
            logger.warning("No articles to save")
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save complete data as JSON (title + content structure)
        json_file = os.path.join(self.output_dir, f"{filename_prefix}_{timestamp}.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        
        # Save human-readable format
        readable_file = os.path.join(self.output_dir, f"{filename_prefix}_readable_{timestamp}.txt")
        with open(readable_file, 'w', encoding='utf-8') as f:
            for i, article in enumerate(articles, 1):
                f.write(f"=== Article {i} ===\n")
                f.write(f"URL: {article['url']}\n")
                f.write(f"Title: {article['title']}\n")
                f.write(f"Content:\n{article['content']}\n")
                f.write("\n" + "="*80 + "\n\n")
        
        # Save just content for training (title + content combined)
        training_file = os.path.join(self.output_dir, f"{filename_prefix}_training_{timestamp}.txt")
        with open(training_file, 'w', encoding='utf-8') as f:
            for article in articles:
                # Combine title and content as raw training data
                f.write(f"{article['title']}\n")
                f.write(f"{article['content']}\n")
                f.write("\n")  # Separator between articles
        
        # Generate statistics
        total_title_chars = sum(len(article['title']) for article in articles)
        total_content_chars = sum(len(article['content']) for article in articles)
        total_chars = total_title_chars + total_content_chars
        
        stats = {
            'total_articles': len(articles),
            'total_characters': total_chars,
            'average_chars_per_article': total_chars / len(articles) if articles else 0,
            'title_chars': total_title_chars,
            'content_chars': total_content_chars,
            'created_at': datetime.now().isoformat(),
            'files_created': [
                f"{filename_prefix}_{timestamp}.json",
                f"{filename_prefix}_readable_{timestamp}.txt",
                f"{filename_prefix}_training_{timestamp}.txt"
            ]
        }
        
        stats_file = os.path.join(self.output_dir, f"{filename_prefix}_stats_{timestamp}.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        # Save scraped URLs to prevent duplicates in future sessions
        self.save_scraped_urls()
        
        logger.info(f"Raw data saved: {len(articles)} articles, {total_chars:,} total characters")
        logger.info(f"Files: {json_file}, {readable_file}, {training_file}")
        
        return timestamp

def main():
    """Main function with user controls for scraping."""
    import sys
    
    scraper = MyAwadyScraper()
    
    print("🇲🇲 Myanmar MyAwady News Scraper")
    print("="*50)
    
    # Interactive mode if no arguments
    if len(sys.argv) == 1:
        print("Interactive mode - Choose your scraping options:")
        print()
        
        # Get user preferences
        print("📰 How many articles do you want to scrape?")
        print("   1. Small test (5 articles)")
        print("   2. Medium collection (25 articles)") 
        print("   3. Large collection (100 articles)")
        print("   4. Custom amount")
        
        choice = input("Enter choice (1-4): ").strip()
        
        if choice == "1":
            max_articles = 5
            max_pages = 1
        elif choice == "2":
            max_articles = 25
            max_pages = 3
        elif choice == "3":
            max_articles = 100
            max_pages = 10
        elif choice == "4":
            try:
                max_articles = int(input("Enter number of articles: "))
                max_pages = max(1, (max_articles // 10) + 1)
            except ValueError:
                print("Invalid input, using default: 25 articles")
                max_articles = 25
                max_pages = 3
        else:
            print("Invalid choice, using default: 25 articles")
            max_articles = 25
            max_pages = 3
        
        print(f"\n⚙️  Configuration:")
        print(f"   - Articles to scrape: {max_articles}")
        print(f"   - Pages to check: {max_pages}")
        print(f"   - Estimated time: {max_articles * 3 // 60 + 1} minutes")
        
        confirm = input(f"\nProceed with scraping? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Scraping cancelled.")
            return
            
    else:
        # Command line arguments
        try:
            max_articles = int(sys.argv[1]) if len(sys.argv) > 1 else 25
            max_pages = int(sys.argv[2]) if len(sys.argv) > 2 else max(1, (max_articles // 10) + 1)
        except ValueError:
            print("Usage: python3 myawady_scraper.py [max_articles] [max_pages]")
            return
    
    # Start scraping
    myawady_url = "https://www.myawady.net.mm/militray"
    print(f"\n🚀 Starting scraping from {myawady_url}")
    print(f"Target: {max_articles} articles from {max_pages} pages")
    
    articles = scraper.scrape_myawady_with_pagination(
        base_url=myawady_url,
        max_articles=max_articles,
        max_pages=max_pages
    )
    
    if articles:
        timestamp = scraper.save_raw_data(articles, "myawady_raw_myanmar")
        print(f"\n🎉 Successfully scraped {len(articles)} articles!")
        print(f"📁 Data saved with timestamp: {timestamp}")
        
        # Print detailed summary  
        total_chars = sum(len(article['title']) + len(article['content']) for article in articles)
        print(f"\n📊 Collection Summary:")
        print(f"   - Articles collected: {len(articles)}")
        print(f"   - Total characters: {total_chars:,}")
        print(f"   - Average characters per article: {total_chars/len(articles):,.0f}")
        
        # Show file locations
        print(f"\n📁 Files created in {scraper.session_dir}/:")
        print(f"   - myawady_raw_myanmar_{timestamp}.json")
        print(f"   - myawady_raw_myanmar_readable_{timestamp}.txt")
        print(f"   - myawady_raw_myanmar_training_{timestamp}.txt")
        print(f"   - myawady_raw_myanmar_stats_{timestamp}.json")
        print(f"\n🔄 Duplicate prevention: scraped_urls.json updated with {len(scraper.visited_urls)} URLs")
        
    else:
        print("❌ No articles were collected")
        print("Possible reasons:")
        print("   - Website is using heavy JavaScript")
        print("   - Network issues")
        print("   - Website structure changed")

if __name__ == "__main__":
    main()