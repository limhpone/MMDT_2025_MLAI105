#!/usr/bin/env python3
"""
Myanmar News Scraper for POS Tagging Data Collection
Focused scraper for DVB and other Myanmar news websites
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
import unicodedata

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DVBScraper:
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
        """Clean Myanmar text and completely remove invisible Unicode characters."""
        if not text:
            return text
        
        # Define characters to completely remove (zero-width and invisible)
        chars_to_remove = [
            '\u200B',  # Zero Width Space
            '\u200C',  # Zero Width Non-Joiner
            '\u200D',  # Zero Width Joiner
            '\u2060',  # Word Joiner
            '\uFEFF',  # Byte Order Mark
            '\u180E',  # Mongolian Vowel Separator
            '\u034F',  # Combining Grapheme Joiner
            '\u2061',  # Function Application
            '\u2062',  # Invisible Times
            '\u2063',  # Invisible Separator
            '\u2064',  # Invisible Plus
            '\u2066',  # Left-to-Right Isolate
            '\u2067',  # Right-to-Left Isolate
            '\u2068',  # First Strong Isolate
            '\u2069',  # Pop Directional Isolate
            '\u202A',  # Left-to-Right Embedding
            '\u202B',  # Right-to-Left Embedding
            '\u202C',  # Pop Directional Formatting
            '\u202D',  # Left-to-Right Override
            '\u202E',  # Right-to-Left Override
        ]
        
        # Define characters to replace with ASCII equivalents
        char_replacements = {
            '\u00A0': ' ',    # Non-breaking space → regular space
            '\u2009': ' ',    # Thin space → regular space
            '\u200A': ' ',    # Hair space → regular space
            '\u2002': ' ',    # En space → regular space
            '\u2003': ' ',    # Em space → regular space
            '\u2004': ' ',    # Three-per-em space → regular space
            '\u2005': ' ',    # Four-per-em space → regular space
            '\u2006': ' ',    # Six-per-em space → regular space
            '\u2007': ' ',    # Figure space → regular space
            '\u2008': ' ',    # Punctuation space → regular space
            '\u205F': ' ',    # Medium mathematical space → regular space
            '\u3000': ' ',    # Ideographic space → regular space
            
            # Typographic quotes → ASCII
            '\u201C': '"',    # Left double quotation mark
            '\u201D': '"',    # Right double quotation mark
            '\u2018': "'",    # Left single quotation mark
            '\u2019': "'",    # Right single quotation mark
            
            # Dashes → ASCII hyphen
            '\u2013': '-',    # En dash
            '\u2014': '-',    # Em dash
            
            # Other symbols
            '\u2026': '...',  # Horizontal ellipsis
        }
        
        # Step 1: Remove invisible characters completely
        for char in chars_to_remove:
            text = text.replace(char, '')
        
        # Step 2: Replace problematic characters with ASCII equivalents
        for old_char, new_char in char_replacements.items():
            text = text.replace(old_char, new_char)
        
        # Step 3: Character-by-character filtering for any remaining problematic chars
        cleaned_chars = []
        for char in text:
            ord_c = ord(char)
            
            # Keep Myanmar script characters (U+1000-U+109F, U+AA60-U+AA7F)
            is_myanmar = (0x1000 <= ord_c <= 0x109F) or (0xAA60 <= ord_c <= 0xAA7F)
            
            # Keep ASCII characters (0-127)
            is_ascii = ord_c < 127
            
            # Keep common safe characters
            is_safe = ord_c in [0x09, 0x0A, 0x0D]  # Tab, newline, carriage return
            
            # Only keep characters that are safe
            if is_ascii or is_myanmar or is_safe:
                cleaned_chars.append(char)
        
        text = ''.join(cleaned_chars)
        
        # Step 4: Normalize whitespace and remove HTML entities
        text = re.sub(r'&[a-zA-Z0-9#]+;', '', text)  # Remove HTML entities
        text = re.sub(r' +', ' ', text)  # Multiple spaces → single space
        text = re.sub(r'\n\s*\n+', '\n\n', text)  # Multiple newlines → double newline
        
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
        """Extract article links from DVB news page."""
        links = []
        
        # Find all links that look like articles
        all_links = soup.find_all('a', href=True)
        
        for link in all_links:
            href = link.get('href')  # type: ignore
            if not href or not isinstance(href, str):
                continue
                
            # Handle both relative and absolute URLs
            if href.startswith('/post/'):
                full_url = urljoin(base_url, href)
            elif '/post/' in href:
                full_url = href
            else:
                continue
                
            # Add unique links (don't filter visited_urls here - done in pagination)
            if full_url not in links:
                links.append(full_url)
        
        return links
    
    def scrape_article(self, url: str) -> Optional[Dict]:
        """Scrape a single DVB article - extract only the actual article content."""
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
        
        # Extract title - simplified approach
        raw_title = ""
        title_selectors = ['h1', '.post-title', '.entry-title', '.article-title', 'title']
        
        for selector in title_selectors:
            title_elem = soup.select_one(selector)
            if title_elem:
                title_text = title_elem.get_text().strip()
                # Simple check - just needs to be Myanmar text and reasonable length
                if title_text and self.is_myanmar_text(title_text) and len(title_text) > 5:
                    raw_title = title_text
                    break
        
        # Extract content - simplified but smart filtering
        raw_content = ""
        
        # Get all paragraphs and filter smartly
        all_paragraphs = soup.find_all('p')
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
        
        # Must have both title and content
        if not raw_title or not raw_content or len(raw_content) < 50:
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
        """Check if text looks like navigation/menu content."""
        # Common navigation indicators (be more specific)
        nav_indicators = [
            'home', 'login', 'english', 'search', 'menu', 
            'follow us', 'contact', 'privacy', 'about dvb'
        ]
        
        text_lower = text.lower()
        
        # Check for navigation patterns
        if any(indicator in text_lower for indicator in nav_indicators):
            return True
        
        # Check for long category-like text (the problematic navigation menu)
        if (len(text) > 200 and len(text.split()) > 30 
            and ('/' in text or '်' in text[:100])):  # Lots of category separators
            return True
        
        # Check for repetitive short phrases (common in menus)
        words = text.split()
        if len(words) > 15 and len(set(words)) < len(words) * 0.5:  # Too much repetition
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
    
    def scrape_dvb_with_pagination(self, base_url: str = "https://burmese.dvb.no/category/8/news", max_articles: int = 100, max_pages: int = 10) -> List[Dict]:
        """Scrape DVB articles with pagination support, skipping already scraped URLs."""
        logger.info(f"Starting DVB scraping from {base_url}")
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
                page_url = f"{base_url}?page={page_num}"
            
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
        
        logger.info(f"DVB scraping completed: {len(all_articles)} NEW articles collected")
        return all_articles
    
    
    def save_raw_data(self, articles: List[Dict], filename_prefix: str = "dvb_raw_myanmar"):
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
    
    scraper = DVBScraper()
    
    print("🇲🇲 Myanmar DVB News Scraper")
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
            print("Usage: python3 myanmar_scraper.py [max_articles] [max_pages]")
            return
    
    # Start scraping
    dvb_url = "https://burmese.dvb.no/category/8/news"
    print(f"\n🚀 Starting scraping from {dvb_url}")
    print(f"Target: {max_articles} articles from {max_pages} pages")
    
    articles = scraper.scrape_dvb_with_pagination(
        base_url=dvb_url,
        max_articles=max_articles,
        max_pages=max_pages
    )
    
    if articles:
        timestamp = scraper.save_raw_data(articles, "dvb_raw_myanmar")
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
        print(f"   - dvb_raw_myanmar_{timestamp}.json")
        print(f"   - dvb_raw_myanmar_readable_{timestamp}.txt")
        print(f"   - dvb_raw_myanmar_training_{timestamp}.txt")
        print(f"   - dvb_raw_myanmar_stats_{timestamp}.json")
        print(f"\n🔄 Duplicate prevention: scraped_urls.json updated with {len(scraper.visited_urls)} URLs")
        
    else:
        print("❌ No articles were collected")
        print("Possible reasons:")
        print("   - Website is using heavy JavaScript")
        print("   - Network issues")
        print("   - Website structure changed")

if __name__ == "__main__":
    main()