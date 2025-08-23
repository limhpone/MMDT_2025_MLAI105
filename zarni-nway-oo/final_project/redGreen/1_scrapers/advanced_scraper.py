#!/usr/bin/env python3
"""
Advanced Myanmar News Scraper with Date Controls
Provides fine-grained control over scraping parameters
"""

from dvb_scraper import DVBScraper
from myawady_scraper import MyAwadyScraper
from khitthit_scraper import KhitthitScraper
from datetime import datetime, timedelta
import argparse
import sys

def parse_date(date_str):
    """Parse date string in various formats."""
    formats = ['%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y', '%d/%m/%Y']
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Invalid date format: {date_str}. Use YYYY-MM-DD")

def create_date_filtered_scraper(scraper_class):
    """Create a scraper with date filtering capabilities."""
    
    class DateFilteredScraper(scraper_class):
        def __init__(self, start_date=None, end_date=None, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.start_date = start_date
            self.end_date = end_date
        
        def scrape_article_with_date_check(self, url: str):
            """Scrape article and check if it falls within date range."""
            article = self.scrape_article(url)
            
            if not article:
                return None
            
            # For now, since we don't extract dates from articles,
            # we'll accept all articles. In a real implementation,
            # you'd extract the article date and filter here.
            
            return article
    
    return DateFilteredScraper

def get_scraper_config(source):
    """Get scraper configuration for different news sources."""
    configs = {
        'dvb': {
            'class': DVBScraper,
            'url': 'https://burmese.dvb.no/category/8/news',
            'name': 'DVB Myanmar',
            'method': 'scrape_dvb_with_pagination'
        },
        'myawady': {
            'class': MyAwadyScraper,
            'url': 'https://www.myawady.net.mm/militray',
            'name': 'MyAwady Military News',
            'method': 'scrape_myawady_with_pagination'
        },
        'khitthit': {
            'class': KhitthitScraper,
            'url': 'https://yktnews.com/category/politics/',
            'name': 'Khitthit Politics News',
            'method': 'scrape_khitthit_with_pagination'
        }
    }
    return configs.get(source.lower())

def main():
    parser = argparse.ArgumentParser(
        description="Advanced Myanmar Multi-Source News Scraper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (DEFAULT - no arguments)
  python3 advanced_scraper.py

  # Scrape from DVB
  python3 advanced_scraper.py --source dvb --articles 50 --pages 5

  # Scrape from MyAwady
  python3 advanced_scraper.py --source myawady --articles 25

  # Scrape from Khitthit
  python3 advanced_scraper.py --source khitthit --articles 30

  # Custom output name
  python3 advanced_scraper.py --source dvb --articles 20 --output my_collection

  # Force interactive even with arguments
  python3 advanced_scraper.py --source dvb --articles 50 --interactive
        """
    )
    
    parser.add_argument('--source', '-s', type=str, choices=['dvb', 'myawady', 'khitthit'], default='dvb',
                       help='News source to scrape (default: dvb)')
    parser.add_argument('--articles', '-a', type=int,
                       help='Number of articles to scrape')
    parser.add_argument('--pages', '-p', type=int, 
                       help='Number of pages to check (auto-calculated if not specified)')
    parser.add_argument('--output', '-o', type=str,
                       help='Output filename prefix (auto-generated if not specified)')
    parser.add_argument('--days-back', type=int,
                       help='Scrape articles from last N days (conceptual)')
    parser.add_argument('--start-date', type=str,
                       help='Start date (YYYY-MM-DD) - conceptual feature')
    parser.add_argument('--end-date', type=str,
                       help='End date (YYYY-MM-DD) - conceptual feature')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Force interactive mode')
    parser.add_argument('--url', type=str,
                       help='Base URL to scrape (overrides source default)')
    parser.add_argument('--delay', type=float, default=2.5,
                       help='Average delay between requests in seconds (default: 2.5)')
    
    args = parser.parse_args()
    
    # Get scraper configuration
    scraper_config = get_scraper_config(args.source)
    if not scraper_config:
        print(f"❌ Unknown news source: {args.source}")
        print("Available sources: dvb, myawady, khitthit")
        return
    
    # Set defaults based on source
    if not args.output:
        args.output = f"{args.source}_raw_myanmar"
    if not args.url:
        args.url = scraper_config['url']
    
    # Force interactive mode if no articles specified OR interactive flag is set
    interactive_mode = args.interactive or args.articles is None
    
    print(f"🇲🇲 Advanced Myanmar News Scraper - {scraper_config['name']}")
    print("="*60)
    
    # Interactive mode
    if interactive_mode:
        print(f"Interactive mode for {scraper_config['name']} - Choose your scraping options:")
        print()
        
        # News source selection if not provided
        if not args.source or interactive_mode:
            print("📺 Choose news source:")
            print("   1. DVB Myanmar (General news)")
            print("   2. MyAwady (Military news)")
            print("   3. Khitthit (Politics news)")
            source_choice = input("Enter choice (1-3): ").strip()
            
            if source_choice == "1":
                args.source = 'dvb'
            elif source_choice == "2":
                args.source = 'myawady'
            elif source_choice == "3":
                args.source = 'khitthit'
            else:
                print("Invalid choice, using DVB")
                args.source = 'dvb'
            
            # Refresh config after selection
            scraper_config = get_scraper_config(args.source)
            if not scraper_config:
                print(f"❌ Unknown news source: {args.source}")
                return
            args.output = f"{args.source}_raw_myanmar"
            args.url = scraper_config['url']
            
            print(f"\n✓ Selected: {scraper_config['name']}\n")
        print()
        
        # Get user preferences
        print(f"📰 How many articles do you want to scrape from {scraper_config['name']}?")
        print("   1. Small test (5 articles)")
        print("   2. Medium collection (25 articles)") 
        print("   3. Large collection (100 articles)")
        print("   4. Custom amount")
        
        choice = input("Enter choice (1-4): ").strip()
        
        if choice == "1":
            args.articles = 5
            max_pages = 1
        elif choice == "2":
            args.articles = 25
            max_pages = 3
        elif choice == "3":
            args.articles = 100
            max_pages = 10
        elif choice == "4":
            try:
                args.articles = int(input("Enter number of articles: "))
                max_pages = max(1, (args.articles // 10) + 1)
            except ValueError:
                print("Invalid input, using default: 25 articles")
                args.articles = 25
                max_pages = 3
        else:
            print("Invalid choice, using default: 25 articles")
            args.articles = 25
            max_pages = 3
    else:
        # Direct mode - use provided arguments
        max_pages = args.pages if args.pages else max(1, (args.articles // 10) + 1)
    
    # Handle date filtering (conceptual for now)
    start_date = None
    end_date = None
    
    if args.days_back:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days_back)
        print(f"📅 Date filter: Last {args.days_back} days")
        print(f"   From: {start_date.strftime('%Y-%m-%d')}")
        print(f"   To: {end_date.strftime('%Y-%m-%d')}")
        print("   ⚠️  Note: Date filtering requires article date extraction (not yet implemented)")
    
    if args.start_date:
        try:
            start_date = parse_date(args.start_date)
            print(f"📅 Start date: {start_date.strftime('%Y-%m-%d')}")
        except ValueError as e:
            print(f"❌ {e}")
            return
    
    if args.end_date:
        try:
            end_date = parse_date(args.end_date)
            print(f"📅 End date: {end_date.strftime('%Y-%m-%d')}")
        except ValueError as e:
            print(f"❌ {e}")
            return
    
    # Show configuration and ask for confirmation
    if interactive_mode or args.days_back or args.start_date or args.end_date:
        print(f"\n⚙️  Scraping Configuration:")
        print(f"   - URL: {args.url}")
        print(f"   - Articles: {args.articles}")
        print(f"   - Pages: {max_pages}")
        print(f"   - Output prefix: {args.output}")
        print(f"   - Delay: {args.delay}s between requests")
        print(f"   - Estimated time: {int(args.articles * args.delay / 60) + 1} minutes")
        
        if start_date or end_date:
            print(f"   - Date filtering: {'Enabled' if start_date or end_date else 'Disabled'}")
        
        confirm = input(f"\n🚀 Proceed with scraping? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Scraping cancelled.")
            return
    
    # Create scraper
    if not scraper_config:
        print(f"❌ Unknown news source: {args.source}")
        return
        
    base_scraper_class = scraper_config['class']
    scraper_class = create_date_filtered_scraper(base_scraper_class)
    scraper = scraper_class(start_date=start_date, end_date=end_date)
    
    print(f"\n🚀 Starting scraping from {args.url}")
    print(f"Target: {args.articles} articles from {max_pages} pages")
    
    # Start scraping using the appropriate method
    scraper_method = getattr(scraper, scraper_config['method'])
    articles = scraper_method(
        base_url=args.url,
        max_articles=args.articles,
        max_pages=max_pages
    )
    
    if articles:
        timestamp = scraper.save_raw_data(articles, args.output)
        print(f"\n🎉 Successfully scraped {len(articles)} articles!")
        print(f"📁 Data saved with timestamp: {timestamp}")
        
        # Print detailed summary
        total_chars = sum(len(article['title']) + len(article['content']) for article in articles)
        
        print(f"\n📊 Collection Summary:")
        print(f"   - Articles collected: {len(articles)}")
        print(f"   - Total characters: {total_chars:,}")
        print(f"   - Average characters per article: {total_chars/len(articles):,.0f}")
        
        # Show sample titles
        print(f"\n📰 Sample article titles:")
        for i, article in enumerate(articles[:3], 1):
            title = article['title'][:60] + "..." if len(article['title']) > 60 else article['title']
            print(f"   {i}. {title}")
        
        # Show file locations
        print(f"\n📁 Files created in {scraper.session_dir}/:")
        print(f"   - {args.output}_{timestamp}.json")
        print(f"   - {args.output}_readable_{timestamp}.txt")
        print(f"   - {args.output}_training_{timestamp}.txt")
        print(f"   - {args.output}_stats_{timestamp}.json")
        print(f"\n🔄 Duplicate prevention: scraped_urls.json updated with {len(scraper.visited_urls)} URLs")
        
    else:
        print("❌ No articles were collected")
        print("Possible solutions:")
        print("   - Try a smaller number of articles")
        print("   - Check your internet connection")
        print("   - Verify the URL is accessible")

if __name__ == "__main__":
    main()