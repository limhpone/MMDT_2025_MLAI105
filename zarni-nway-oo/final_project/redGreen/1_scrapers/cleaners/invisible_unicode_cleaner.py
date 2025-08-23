#!/usr/bin/env python3
"""
Comprehensive Invisible Unicode Character Cleaner
Cleans already scraped data files by removing invisible/problematic Unicode characters
"""

import json
import sys
import os
import re
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import unicodedata


class InvisibleUnicodeCleaner:
    def __init__(self):
        # Comprehensive list of invisible characters (from myawady + additional)
        self.invisible_chars = {
            # From myawady scraper (known problematic)
            '\u200B': 'Zero Width Space',
            '\u200C': 'Zero Width Non-Joiner',
            '\u200D': 'Zero Width Joiner',
            '\u2060': 'Word Joiner',
            '\uFEFF': 'Byte Order Mark',
            '\u00A0': 'Non-breaking Space',  # Main culprit in myawady
            '\u2009': 'Thin Space',
            '\u200A': 'Hair Space',
            '\u2028': 'Line Separator',
            '\u2029': 'Paragraph Separator',
            
            # Additional invisible characters
            '\u180E': 'Mongolian Vowel Separator',
            '\u034F': 'Combining Grapheme Joiner',
            '\u2061': 'Function Application',
            '\u2062': 'Invisible Times',
            '\u2063': 'Invisible Separator',
            '\u2064': 'Invisible Plus',
            '\u115F': 'Hangul Choseong Filler',
            '\u1160': 'Hangul Jungseong Filler',
            '\u17B4': 'Khmer Vowel Inherent Aq',
            '\u17B5': 'Khmer Vowel Inherent Aa',
            '\u3164': 'Hangul Filler',
            '\uFFA0': 'Halfwidth Hangul Filler',
            
            # Additional control characters often problematic in web scraping
            '\u00AD': 'Soft Hyphen',
            '\u061C': 'Arabic Letter Mark',
            '\u2066': 'Left-to-Right Isolate',
            '\u2067': 'Right-to-Left Isolate',
            '\u2068': 'First Strong Isolate',
            '\u2069': 'Pop Directional Isolate',
            '\u202A': 'Left-to-Right Embedding',
            '\u202B': 'Right-to-Left Embedding',
            '\u202C': 'Pop Directional Formatting',
            '\u202D': 'Left-to-Right Override',
            '\u202E': 'Right-to-Left Override',
            '\u2002': 'En Space',
            '\u2003': 'Em Space',
            '\u2004': 'Three-Per-Em Space',
            '\u2005': 'Four-Per-Em Space',
            '\u2006': 'Six-Per-Em Space',
            '\u2007': 'Figure Space',
            '\u2008': 'Punctuation Space',
            '\u200B': 'Zero Width Space',
            '\u205F': 'Medium Mathematical Space',
            '\u3000': 'Ideographic Space',
            
            # Typographic quotes and punctuation (from web content)
            '\u201C': 'Left Double Quotation Mark',  # " 
            '\u201D': 'Right Double Quotation Mark', # "
            '\u2018': 'Left Single Quotation Mark',  # '
            '\u2019': 'Right Single Quotation Mark', # '
            '\u2013': 'En Dash',                     # –
            '\u2014': 'Em Dash',                     # —
            '\u2026': 'Horizontal Ellipsis',         # …
        }
        
        # Statistics tracking
        self.cleaning_stats = {
            'files_processed': 0,
            'articles_processed': 0,
            'articles_cleaned': 0,
            'characters_removed': 0,
            'character_breakdown': {},
            'errors': []
        }
    
    def clean_text(self, text: str) -> Tuple[str, Dict[str, int]]:
        """Clean text by removing invisible Unicode characters and normalizing typography."""
        if not text:
            return text, {}
        
        cleaned_text = text
        chars_removed = {}
        
        # Define replacement mapping for typographic characters
        typographic_replacements = {
            '\u201C': '"',  # Left double quotation mark → regular quote
            '\u201D': '"',  # Right double quotation mark → regular quote
            '\u2018': "'",  # Left single quotation mark → regular apostrophe
            '\u2019': "'",  # Right single quotation mark → regular apostrophe
            '\u2013': '-',  # En dash → regular hyphen
            '\u2014': '-',  # Em dash → regular hyphen  
            '\u2026': '...', # Horizontal ellipsis → three dots
        }
        
        # Handle each invisible character type and count occurrences
        for char, description in self.invisible_chars.items():
            count = cleaned_text.count(char)
            if count > 0:
                chars_removed[f'{char} (U+{ord(char):04X}) - {description}'] = count
                
                # Use appropriate replacement
                if char in typographic_replacements:
                    cleaned_text = cleaned_text.replace(char, typographic_replacements[char])
                else:
                    # For truly invisible characters, remove them completely (don't replace with space)
                    if char in ['\u200B', '\u200C', '\u200D', '\u2060', '\uFEFF', '\u180E', '\u034F', 
                               '\u2061', '\u2062', '\u2063', '\u2064', '\u2066', '\u2067', '\u2068', '\u2069',
                               '\u202A', '\u202B', '\u202C', '\u202D', '\u202E']:
                        cleaned_text = cleaned_text.replace(char, '')
                    else:
                        # For spaces, replace with regular space
                        cleaned_text = cleaned_text.replace(char, ' ')
        
        # Additional cleaning: remove problematic control/format characters but preserve Myanmar
        extra_removed = 0
        filtered_chars = []
        for c in cleaned_text:
            category = unicodedata.category(c)
            ord_c = ord(c)
            
            # Only remove control/format characters that are NOT Myanmar script
            # Myanmar script ranges: U+1000-U+109F, U+AA60-U+AA7F
            is_myanmar = (0x1000 <= ord_c <= 0x109F) or (0xAA60 <= ord_c <= 0xAA7F)
            
            if category in ['Cf', 'Cc'] and ord_c > 127 and not is_myanmar:
                extra_removed += 1
            else:
                filtered_chars.append(c)
        
        if extra_removed > 0:
            chars_removed['Other Control Characters (non-Myanmar)'] = extra_removed
            cleaned_text = ''.join(filtered_chars)
        
        # Normalize whitespace (replace multiple spaces with single space)
        cleaned_text = re.sub(r' +', ' ', cleaned_text.strip())
        
        # Remove empty lines and normalize line breaks
        cleaned_text = re.sub(r'\n\s*\n+', '\n\n', cleaned_text)
        cleaned_text = cleaned_text.strip()
        
        return cleaned_text, chars_removed
    
    def clean_article(self, article: Dict) -> Tuple[Dict, bool]:
        """Clean a single article and return cleaned version + whether changes were made."""
        cleaned_article = article.copy()
        changes_made = False
        article_char_stats = {}
        
        # Clean title
        if 'title' in article and article['title']:
            cleaned_title, title_chars = self.clean_text(article['title'])
            # Always update title even if no chars detected (might have other cleaning)
            cleaned_article['title'] = cleaned_title
            if title_chars:
                changes_made = True
                article_char_stats.update(title_chars)
        
        # Clean content
        if 'content' in article and article['content']:
            cleaned_content, content_chars = self.clean_text(article['content'])
            # Always update content even if no chars detected (might have other cleaning)
            cleaned_article['content'] = cleaned_content
            if content_chars:
                changes_made = True
                for char_type, count in content_chars.items():
                    article_char_stats[char_type] = article_char_stats.get(char_type, 0) + count
        
        # Update cleaning metadata
        if changes_made:
            cleaned_article['cleaned_at'] = datetime.now().isoformat()
            cleaned_article['invisible_chars_removed'] = article_char_stats
        
        return cleaned_article, changes_made
    
    def clean_json_file(self, file_path: str, output_path: Optional[str] = None, 
                       create_backup: bool = True) -> Dict:
        """Clean a JSON file containing scraped articles."""
        print(f"🧹 Cleaning file: {file_path}")
        
        # Create backup if requested
        if create_backup:
            backup_path = f"{file_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            try:
                shutil.copy2(file_path, backup_path)
                print(f"💾 Backup created: {backup_path}")
            except Exception as e:
                print(f"⚠️  Warning: Could not create backup: {e}")
        
        # Load the file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                articles = json.load(f)
        except Exception as e:
            error_msg = f"Error reading {file_path}: {e}"
            self.cleaning_stats['errors'].append(error_msg)
            print(f"❌ {error_msg}")
            return {'success': False, 'error': str(e)}
        
        if not isinstance(articles, list):
            error_msg = f"File {file_path} does not contain a list of articles"
            self.cleaning_stats['errors'].append(error_msg)
            print(f"❌ {error_msg}")
            return {'success': False, 'error': 'Invalid file format'}
        
        print(f"📊 Processing {len(articles)} articles...")
        
        # Clean each article
        cleaned_articles = []
        articles_with_changes = 0
        total_chars_removed = 0
        
        for i, article in enumerate(articles):
            cleaned_article, changes_made = self.clean_article(article)
            cleaned_articles.append(cleaned_article)
            
            if changes_made:
                articles_with_changes += 1
                
                # Update statistics
                if 'invisible_chars_removed' in cleaned_article:
                    for char_type, count in cleaned_article['invisible_chars_removed'].items():
                        self.cleaning_stats['character_breakdown'][char_type] = \
                            self.cleaning_stats['character_breakdown'].get(char_type, 0) + count
                        total_chars_removed += count
        
        # Determine output path
        if output_path is None:
            # Add _cleaned suffix to original filename
            base, ext = os.path.splitext(file_path)
            output_path = f"{base}_cleaned{ext}"
        
        # Save cleaned file
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(cleaned_articles, f, ensure_ascii=False, indent=2)
        except Exception as e:
            error_msg = f"Error saving cleaned file {output_path}: {e}"
            self.cleaning_stats['errors'].append(error_msg)
            print(f"❌ {error_msg}")
            return {'success': False, 'error': str(e)}
        
        # Update overall statistics
        self.cleaning_stats['files_processed'] += 1
        self.cleaning_stats['articles_processed'] += len(articles)
        self.cleaning_stats['articles_cleaned'] += articles_with_changes
        self.cleaning_stats['characters_removed'] += total_chars_removed
        
        result = {
            'success': True,
            'input_file': file_path,
            'output_file': output_path,
            'total_articles': len(articles),
            'articles_cleaned': articles_with_changes,
            'characters_removed': total_chars_removed,
            'backup_created': backup_path if create_backup else None
        }
        
        print(f"✅ Cleaned {articles_with_changes}/{len(articles)} articles, removed {total_chars_removed} characters")
        print(f"📄 Cleaned file saved as: {output_path}")
        
        return result
    
    def clean_multiple_files(self, file_paths: List[str], output_dir: Optional[str] = None, 
                            create_backup: bool = True) -> List[Dict]:
        """Clean multiple JSON files."""
        print(f"🚀 Starting batch cleaning of {len(file_paths)} files...")
        print()
        
        results = []
        
        for file_path in file_paths:
            if not os.path.exists(file_path):
                error_msg = f"File not found: {file_path}"
                self.cleaning_stats['errors'].append(error_msg)
                print(f"❌ {error_msg}")
                continue
            
            # Determine output path
            output_path = None
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                filename = os.path.basename(file_path)
                base, ext = os.path.splitext(filename)
                output_path = os.path.join(output_dir, f"{base}_cleaned{ext}")
            
            result = self.clean_json_file(file_path, output_path, create_backup)
            results.append(result)
            print()  # Add spacing between files
        
        return results
    
    def generate_cleaning_report(self, results: List[Dict]) -> str:
        """Generate a comprehensive cleaning report."""
        report = []
        report.append("🧹 INVISIBLE UNICODE CHARACTER CLEANING REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Overall summary
        successful_files = [r for r in results if r.get('success', False)]
        failed_files = [r for r in results if not r.get('success', False)]
        
        total_articles = sum(r.get('total_articles', 0) for r in successful_files)
        total_cleaned = sum(r.get('articles_cleaned', 0) for r in successful_files)
        total_chars_removed = sum(r.get('characters_removed', 0) for r in successful_files)
        
        report.append("📊 SUMMARY")
        report.append(f"   Files processed: {len(successful_files)}/{len(results)}")
        report.append(f"   Total articles: {total_articles:,}")
        report.append(f"   Articles cleaned: {total_cleaned:,}")
        report.append(f"   Clean articles: {total_articles - total_cleaned:,}")
        report.append(f"   Total characters removed: {total_chars_removed:,}")
        report.append("")
        
        # Character breakdown
        if self.cleaning_stats['character_breakdown']:
            report.append("🔤 CHARACTERS REMOVED BREAKDOWN")
            sorted_chars = sorted(
                self.cleaning_stats['character_breakdown'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            for char_desc, count in sorted_chars:
                report.append(f"   {char_desc}: {count:,} occurrences")
            report.append("")
        
        # File-by-file results
        if successful_files:
            report.append("📄 FILE PROCESSING RESULTS")
            for result in successful_files:
                cleaned_pct = (result['articles_cleaned'] / result['total_articles'] * 100) if result['total_articles'] > 0 else 0
                report.append(f"   ✅ {os.path.basename(result['input_file'])}")
                report.append(f"      Articles: {result['total_articles']}, Cleaned: {result['articles_cleaned']} ({cleaned_pct:.1f}%)")
                report.append(f"      Characters removed: {result['characters_removed']:,}")
                report.append(f"      Output: {os.path.basename(result['output_file'])}")
                if result.get('backup_created'):
                    report.append(f"      Backup: {os.path.basename(result['backup_created'])}")
                report.append("")
        
        # Errors
        if failed_files or self.cleaning_stats['errors']:
            report.append("❌ ERRORS ENCOUNTERED")
            for result in failed_files:
                report.append(f"   {result.get('input_file', 'Unknown file')}: {result.get('error', 'Unknown error')}")
            for error in self.cleaning_stats['errors']:
                report.append(f"   {error}")
            report.append("")
        
        # Recommendations
        report.append("💡 RECOMMENDATIONS")
        if total_cleaned > 0:
            report.append(f"   ✅ {total_cleaned:,} articles have been cleaned of invisible characters")
            report.append(f"   📊 Quality improved: removed {total_chars_removed:,} problematic characters")
            report.append("   🔄 Consider re-running sentiment analysis on cleaned data")
            report.append("   🧪 Test cleaned data for improved processing quality")
        else:
            report.append("   ✅ All analyzed text was already clean!")
            report.append("   📊 No invisible Unicode characters found.")
        
        if self.cleaning_stats['errors']:
            report.append("   ⚠️  Review and fix errors listed above")
        
        return "\n".join(report)


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 invisible_unicode_cleaner.py <file1.json> [file2.json] ...")
        print("       python3 invisible_unicode_cleaner.py --output-dir <dir> <file1.json> [file2.json] ...")
        print("")
        print("Examples:")
        print("  # Clean single file (creates backup and _cleaned version)")
        print("  python3 invisible_unicode_cleaner.py data/raw_data/session_*/dvb_*.json")
        print("")
        print("  # Clean multiple files to specific output directory")
        print("  python3 invisible_unicode_cleaner.py --output-dir cleaned_data/ data/**/*.json")
        print("")
        print("Options:")
        print("  --output-dir DIR  Save cleaned files to specified directory")
        print("  --no-backup      Don't create backup files")
        sys.exit(1)
    
    # Parse arguments
    output_dir = None
    create_backup = True
    file_paths = []
    
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '--output-dir' and i + 1 < len(sys.argv):
            output_dir = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--no-backup':
            create_backup = False
            i += 1
        else:
            file_paths.append(sys.argv[i])
            i += 1
    
    if not file_paths:
        print("❌ No files specified for cleaning")
        sys.exit(1)
    
    print("🧹 Starting invisible Unicode character cleaning...")
    print(f"📁 Files to clean: {len(file_paths)}")
    if output_dir:
        print(f"📂 Output directory: {output_dir}")
    if not create_backup:
        print("⚠️  Backup creation disabled")
    print()
    
    # Initialize cleaner and process files
    cleaner = InvisibleUnicodeCleaner()
    results = cleaner.clean_multiple_files(file_paths, output_dir, create_backup)
    
    # Generate and save report
    report = cleaner.generate_cleaning_report(results)
    print("\n" + "="*60)
    print(report)
    
    # Save report to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    report_file = f"logs/cleaning_report_{timestamp}.txt"
    
    try:
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n💾 Detailed report saved to: {report_file}")
    except Exception as e:
        print(f"⚠️  Could not save report: {e}")
    
    # Summary
    successful = sum(1 for r in results if r.get('success', False))
    total_articles_cleaned = sum(r.get('articles_cleaned', 0) for r in results if r.get('success', False))
    total_chars_removed = sum(r.get('characters_removed', 0) for r in results if r.get('success', False))
    
    print(f"\n🎉 CLEANING COMPLETE!")
    print(f"   ✅ {successful}/{len(file_paths)} files processed successfully")
    print(f"   🧹 {total_articles_cleaned:,} articles cleaned")
    print(f"   🗑️  {total_chars_removed:,} invisible characters removed")


if __name__ == "__main__":
    main()