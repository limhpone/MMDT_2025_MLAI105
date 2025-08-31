#!/usr/bin/env python3
"""
Myanmar News Scraper for Khitthit News Website
Focused scraper for Khitthit politics category
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

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class KhitthitScraper:
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
        """Clean Myanmar text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        # Remove HTML entities
        text = re.sub(r'&[a-zA-Z0-9#]+;', '', text)
        # Keep Myanmar script, numbers, and basic punctuation
        text = re.sub(r'[^\u1000-\u109F\u1040-\u1049\uAA60-\uAA7F\s\w.,!?()-]', '', text)
        return text.strip()
    
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
        """Extract article links from Khitthit news page."""
        links = []
        
        # Find all links that match Khitthit pattern /YYYY/MM/number/
        all_links = soup.find_all('a', href=True)
        
        for link in all_links:
            href = link.get('href')  # type: ignore
            if not href or not isinstance(href, str):
                continue
                
            # Handle both relative and absolute URLs for Khitthit /YYYY/MM/number/ pattern
            khitthit_pattern = re.match(r'.*/(\d{4})/(\d{2})/(\d+)/?$', href)
            if khitthit_pattern:
                if href.startswith('/'):
                    full_url = urljoin(base_url, href)
                elif 'yktnews.com' in href:
                    full_url = href
                else:
                    continue
                    
                # Add unique links
                if full_url not in links:
                    links.append(full_url)
        
        return links
    
    def scrape_article(self, url: str) -> Optional[Dict]:
        """Scrape a single Khitthit article - extract only the actual article content."""
        soup = self.get_page(url)
        if not soup:
            return None
        
        
        # Remove unwanted elements - but be more conservative to avoid removing article content
        unwanted_selectors = [
            'script', 'style', 'nav', 'header', 'footer'
        ]
        
        for selector in unwanted_selectors:
            for element in soup.select(selector):
                element.decompose()
        
        # Extract title - try Khitthit specific selectors first
        raw_title = ""
        title_selectors = [
            '.tdb-title-text',  # Khitthit specific
            'h1.entry-title', 
            'h1', 
            '.post-title', 
            '.entry-title', 
            '.article-title',
            'title'
        ]
        
        for selector in title_selectors:
            title_elem = soup.select_one(selector)
            if title_elem:
                title_text = title_elem.get_text().strip()
                # Simple check - just needs to be Myanmar text and reasonable length
                if title_text and self.is_myanmar_text(title_text) and len(title_text) > 5:
                    raw_title = title_text
                    break
        
        # Extract content - try multiple strategies for Khitthit
        raw_content = ""
        
        # Strategy 1: Look for Khitthit main content container and extract ALL paragraphs
        main_content_elem = soup.select_one('.tdb-block-inner.td-fix-index')
        if main_content_elem:
            paragraphs = main_content_elem.find_all('p', recursive=True)
            article_paragraphs = []
            
            for p in paragraphs:
                p_text = p.get_text().strip()
                if (p_text and self.is_myanmar_text(p_text) and len(p_text) > 15):
                    if not self.is_navigation_text(p_text):
                        article_paragraphs.append(p_text)
            
            if article_paragraphs:
                logger.debug(f"Strategy 1: Found {len(article_paragraphs)} paragraphs")
                raw_content = ' '.join(article_paragraphs)
            else:
                logger.debug("Strategy 1: Found container but no valid paragraphs")
        else:
            logger.debug("Strategy 1: Main content container not found")
        
        # Fallback: Try other selectors if main content didn't work
        if not raw_content:
            logger.debug("Strategy 1 failed, trying fallback selectors")
            content_selectors = [
                '.tdb-single-content',
                '.td-post-content', 
                '.tdb-block-inner',
                '.entry-content',
                '.post-content',
                '.content',
                '.article-content'
            ]
            
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    paragraphs = content_elem.find_all(['p'])
                    article_paragraphs = []
                    
                    for p in paragraphs:
                        p_text = p.get_text().strip()
                        if (p_text and self.is_myanmar_text(p_text) 
                            and len(p_text) > 20
                            and not self.is_navigation_text(p_text)):
                            article_paragraphs.append(p_text)
                    
                    if article_paragraphs:
                        logger.debug(f"Fallback {selector}: Found {len(article_paragraphs)} paragraphs")
                        logger.debug(f"Paragraph lengths: {[len(p) for p in article_paragraphs]}")
                        raw_content = ' '.join(article_paragraphs)
                        
                        # For debugging - let's also check if we missed any paragraphs
                        all_paragraphs = content_elem.find_all(['p'])
                        logger.debug(f"Total paragraphs in {selector}: {len(all_paragraphs)}")
                        missed = len(all_paragraphs) - len(article_paragraphs)
                        if missed > 0:
                            logger.debug(f"Missed {missed} paragraphs - checking what was filtered out")
                            for p in all_paragraphs:
                                p_text = p.get_text().strip()
                                if p_text and len(p_text) > 20:
                                    is_myanmar = self.is_myanmar_text(p_text)
                                    is_nav = self.is_navigation_text(p_text)
                                    accepted = p_text in raw_content
                                    logger.debug(f"P[{len(p_text)}]: Myanmar={is_myanmar}, Nav={is_nav}, Accepted={accepted}")
                                    if not accepted:
                                        logger.debug(f"  Rejected: {p_text[:100]}...")
                        break
        
        # Strategy 2: If no content found, look for all paragraphs and filter carefully
        if not raw_content:
            # Look for both paragraphs and divs
            all_paragraphs = soup.find_all(['p', 'div'])
            article_paragraphs = []
            
            for p in all_paragraphs:
                p_text = p.get_text().strip()
                # Less strict filtering for content
                if (p_text and self.is_myanmar_text(p_text) 
                    and len(p_text) > 30  # Moderate minimum length 
                    and not self.is_navigation_text(p_text)
                    # Check for reasonable content
                    and len(p_text.split()) > 5):  # At least 5 words
                    article_paragraphs.append(p_text)
            
            if article_paragraphs:  # Accept any good paragraphs found
                raw_content = ' '.join(article_paragraphs)
        
        # Strategy 3: More targeted approach - look for actual article content
        if not raw_content:
            # Try to find content in common article containers
            article_containers = [
                '.tdb_single_post_content',
                '.td-post-content',
                '.post-content', 
                '.article-content',
                '[class*="post"]',
                '[class*="article"]'
            ]
            
            for container in article_containers:
                container_elem = soup.select_one(container)
                if container_elem:
                    # Get all text from this container and filter carefully
                    texts = []
                    for elem in container_elem.find_all(['p', 'div']):
                        elem_text = elem.get_text().strip()
                        if (elem_text and self.is_myanmar_text(elem_text) 
                            and len(elem_text) > 30
                            and not self.is_navigation_text(elem_text)):
                            texts.append(elem_text)
                    
                    if texts:
                        raw_content = ' '.join(texts)
                        break
        
        # Strategy 4: Comprehensive content extraction - find all article paragraphs
        if not raw_content:
            logger.debug("Attempting Strategy 4: Comprehensive extraction")
            article_parts = []
            
            for elem in soup.find_all(['p', 'div']):
                elem_text = elem.get_text().strip()
                
                if (elem_text and self.is_myanmar_text(elem_text) 
                    and len(elem_text) > 50
                    and not self.is_navigation_text(elem_text)):
                    
                    has_news_content = any(keyword in elem_text for keyword in [
                        'TNLA', 'စစ်တပ်', 'စစ်ကောင်စီ', 'တရုတ်', 'မြို့နယ်',
                        'ကြောင်း', 'ပြောဆို', 'ထုတ်ဖော်', 'လက်လွှတ်', 'သိမ်းပိုက်',
                        'မဟာမိတ်', 'ဖိအား', 'ဆွေးနွေး', 'ရှမ်း', 'နောင်ချို'
                    ])
                    
                    if has_news_content:
                        logger.debug(f"Strategy 4: Found article part [{len(elem_text)} chars]: {elem_text[:100]}...")
                        article_parts.append(elem_text)
            
            if article_parts:
                logger.debug(f"Strategy 4: Found {len(article_parts)} article parts, total length: {sum(len(p) for p in article_parts)}")
                raw_content = ' '.join(article_parts)
            else:
                logger.debug("Strategy 4: No article parts found")
        
        # DEBUG: Log what we found
        logger.debug(f"Title found: {'Yes' if raw_title else 'No'}")
        logger.debug(f"Content length: {len(raw_content) if raw_content else 0}")
        if raw_content:
            logger.debug(f"Content preview: {raw_content[:100]}...")
        
        # Must have both title and content - more lenient for Khitthit short articles
        if not raw_title or not raw_content or len(raw_content) < 50:
            # DEBUG: Try to show what text we did find
            logger.debug(f"Validation failed: title={bool(raw_title)}, content_length={len(raw_content) if raw_content else 0}")
            if raw_content:
                logger.debug(f"Content found but too short: {raw_content}")
            else:
                logger.debug("No content found at all - checking what text exists")
                sample_text = soup.get_text()[:500]
                logger.debug(f"Sample page text: {sample_text}")
            return None
        
        # Mark URL as successfully scraped
        self.visited_urls.add(url)
        
        return {
            'url': url,
            'title': raw_title,
            'content': raw_content,
            'scraped_at': datetime.now().isoformat()
        }
    
    def is_navigation_text(self, text: str) -> bool:
        """Check if text looks like navigation/menu content - conservative to avoid rejecting articles."""
        
        text_lower = text.lower()
        
        # Only flag obvious navigation patterns
        
        # 1. Social media link collections (strong indicator of navigation)
        social_words = ['Facebook', 'Mail', 'Telegram', 'TikTok', 'Website', 'Youtube']
        social_matches = sum(1 for word in social_words if word in text)
        if social_matches >= 4:  # Need 4+ social media words
            return True
        
        # 2. Navigation menu combinations - only if many menu items together
        khitthit_menu_items = [
            'နေ့စဉ်သတင်း', 'နိုင်ငံရေး သတင်း', 'TV Channel',
            'အယ်ဒီတာ့ အာဘော်', 'အင်တာဗျူး အစီအစဉ်'
        ]
        menu_count = sum(1 for menu_item in khitthit_menu_items if menu_item in text)
        if menu_count >= 3:  # Need 3+ menu items together
            return True
        
        # 3. Common English navigation words
        nav_indicators = ['home', 'login', 'english', 'search', 'menu', 'follow us', 'contact']
        nav_matches = sum(1 for indicator in nav_indicators if indicator in text_lower)
        if nav_matches >= 2:  # Need multiple nav indicators
            return True
        
        # 4. Very repetitive text (characteristic of menus)
        words = text.split()
        if len(words) > 25 and len(set(words)) < len(words) * 0.3:  # Less than 30% unique words
            return True
        
        # 5. Text that's mostly non-Myanmar characters (likely English navigation)
        myanmar_chars = len([c for c in text if '\u1000' <= c <= '\u109F'])
        total_chars = len([c for c in text if not c.isspace()])
        if total_chars > 20 and myanmar_chars / total_chars < 0.1:  # Less than 10% Myanmar
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
    
    def scrape_khitthit_with_pagination(self, base_url: str = "https://yktnews.com/category/politics/", max_articles: int = 100, max_pages: int = 10) -> List[Dict]:
        """Scrape Khitthit articles with pagination support, skipping already scraped URLs."""
        logger.info(f"Starting Khitthit scraping from {base_url}")
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
                # Khitthit/WordPress pagination format  
                page_url = f"{base_url}page/{page_num}/"
            
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
        
        logger.info(f"Collection summary: {len(new_article_links)} new links, {duplicates_found} duplicates skipped")
        
        # Now scrape the NEW articles only
        articles_to_scrape = new_article_links[:max_articles]
        for i, link in enumerate(articles_to_scrape, 1):
            logger.info(f"Scraping NEW article {i}/{len(articles_to_scrape)}: {link}")
            
            article = self.scrape_article(link)
            if article:
                all_articles.append(article)
                content_length = len(article['content'])
                logger.info(f"✓ Scraped: {article['title'][:60]}... ({content_length} characters)")
            else:
                logger.warning(f"✗ Failed to scrape: {link}")
            
            # Respectful delay between articles
            time.sleep(random.uniform(2, 4))
        
        logger.info(f"Khitthit scraping completed: {len(all_articles)} NEW articles collected")
        return all_articles
    
    
    def save_raw_data(self, articles: List[Dict], filename_prefix: str = "khitthit_raw_myanmar"):
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
    
    scraper = KhitthitScraper()
    
    print("🇲🇲 Myanmar Khitthit News Scraper")
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
            max_pages = 3  # Start with 3 pages, will extend automatically if needed
        elif choice == "2":
            max_articles = 25
            max_pages = 5  # Start with 5 pages, will extend automatically if needed
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
        print(f"   - Initial pages to check: {max_pages} (will extend automatically if needed)")
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
            print("Usage: python3 khitthit_scraper.py [max_articles] [max_pages]")
            return
    
    # Start scraping
    khitthit_url = "https://yktnews.com/category/politics/"
    print(f"\n🚀 Starting scraping from {khitthit_url}")
    print(f"Target: {max_articles} articles from {max_pages} pages")
    
    articles = scraper.scrape_khitthit_with_pagination(
        base_url=khitthit_url,
        max_articles=max_articles,
        max_pages=max_pages
    )
    
    if articles:
        timestamp = scraper.save_raw_data(articles, "khitthit_raw_myanmar")
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
        print(f"   - khitthit_raw_myanmar_{timestamp}.json")
        print(f"   - khitthit_raw_myanmar_readable_{timestamp}.txt")
        print(f"   - khitthit_raw_myanmar_training_{timestamp}.txt")
        print(f"   - khitthit_raw_myanmar_stats_{timestamp}.json")
        print(f"\n🔄 Duplicate prevention: scraped_urls.json updated with {len(scraper.visited_urls)} URLs")
        
    else:
        print("❌ No articles were collected")
        print("Possible reasons:")
        print("   - Website is using heavy JavaScript")
        print("   - Network issues")
        print("   - Website structure changed")

if __name__ == "__main__":
    main()