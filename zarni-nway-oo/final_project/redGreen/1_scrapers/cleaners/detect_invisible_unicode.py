#!/usr/bin/env python3
"""
Invisible Unicode Character Detection Script
Analyzes scraped data files to identify invisible/problematic Unicode characters
"""

import json
import sys
from collections import defaultdict
from typing import Dict, List
import unicodedata


class InvisibleUnicodeDetector:
    def __init__(self):
        # Known problematic invisible characters from myawady scraper
        self.known_invisible_chars = {
            '\u200B': 'Zero Width Space',
            '\u200C': 'Zero Width Non-Joiner',
            '\u200D': 'Zero Width Joiner',
            '\u2060': 'Word Joiner',
            '\uFEFF': 'Byte Order Mark',
            '\u00A0': 'Non-breaking Space',
            '\u2009': 'Thin Space',
            '\u200A': 'Hair Space',
            '\u2028': 'Line Separator',
            '\u2029': 'Paragraph Separator',
        }
        
        # Additional characters to check for
        self.additional_suspects = {
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
        }
        
        # Combine all suspicious characters
        self.all_suspects = {**self.known_invisible_chars, **self.additional_suspects}
        
        # Results storage
        self.detection_results = defaultdict(lambda: defaultdict(int))
        self.affected_articles = defaultdict(list)
        self.char_positions = defaultdict(list)
        
    def detect_invisible_chars_in_text(self, text: str, article_id: str = "") -> Dict[str, int]:
        """Detect invisible Unicode characters in a text string with comprehensive analysis."""
        found_chars = {}
        
        for i, char in enumerate(text):
            ord_c = ord(char)
            
            # Check if character is in Myanmar script ranges
            is_myanmar = (0x1000 <= ord_c <= 0x109F) or (0xAA60 <= ord_c <= 0xAA7F)
            
            # Skip legitimate Myanmar script characters
            if is_myanmar:
                continue
                
            category = unicodedata.category(char)
            
            # Detect various types of problematic characters
            is_problematic = False
            reason = ""
            
            # 1. Known problematic characters from our list
            if char in self.all_suspects:
                is_problematic = True
                reason = self.all_suspects[char]
            
            # 2. Zero-width and invisible formatting characters
            elif category == 'Cf':  # Format characters
                is_problematic = True
                reason = f"Format Character [Cat: {category}]"
            
            # 3. Control characters (but allow common ones like \n, \t, \r)
            elif category == 'Cc' and ord_c > 127:  # Non-ASCII control
                is_problematic = True
                reason = f"Control Character [Cat: {category}]"
            
            # 4. Non-standard spaces (but allow regular space U+0020)
            elif category in ['Zs', 'Zl', 'Zp'] and ord_c != 0x20:  # Not regular space
                is_problematic = True
                reason = f"Non-standard Space [Cat: {category}]"
            
            # 5. Combining marks that are not Myanmar (could be problematic)
            elif category in ['Mn', 'Mc', 'Me'] and ord_c > 127:  # Non-ASCII combining marks
                is_problematic = True
                reason = f"Non-Myanmar Combining Mark [Cat: {category}]"
            
            # 6. Special detection for common invisible characters that might be missed
            elif ord_c in [
                0x00AD,  # Soft hyphen
                0x034F,  # Combining grapheme joiner
                0x061C,  # Arabic letter mark
                0x180E,  # Mongolian vowel separator
                0x2060,  # Word joiner
                0x2066, 0x2067, 0x2068, 0x2069,  # Directional isolates
                0x202A, 0x202B, 0x202C, 0x202D, 0x202E,  # Directional formatting
                0xFEFF,  # Zero width no-break space / BOM
            ]:
                is_problematic = True
                reason = "Known Invisible Character"
            
            # 7. Characters that render as whitespace but aren't regular space
            elif char.isspace() and ord_c != 0x20 and ord_c != 0x09 and ord_c != 0x0A and ord_c != 0x0D:
                is_problematic = True
                reason = "Non-standard Whitespace"
            
            if is_problematic:
                try:
                    char_name = unicodedata.name(char, f'UNNAMED-{ord_c:04X}')
                    unicode_point = f'U+{ord_c:04X}'
                    
                    # Make character visible in output
                    if ord_c < 32 or category in ['Cf', 'Cc']:
                        visible_char = f'[U+{ord_c:04X}]'  # Show as code point if truly invisible
                    else:
                        visible_char = repr(char)  # Show quoted version
                    
                    key = f'{visible_char} ({unicode_point}) - {char_name} - {reason}'
                    
                    found_chars[key] = found_chars.get(key, 0) + 1
                    
                    if article_id:
                        self.char_positions[key].append({
                            'article_id': article_id,
                            'position': i,
                            'context_before': repr(text[max(0, i-10):i]),
                            'context_after': repr(text[i+1:i+11]),
                            'char_hex': f'0x{ord_c:04X}',
                            'char_decimal': str(ord_c)
                        })
                except:
                    # Fallback for characters that can't be named
                    key = f'[U+{ord_c:04X}] - UNKNOWN CHARACTER - {reason}'
                    found_chars[key] = found_chars.get(key, 0) + 1
        
        return found_chars
    
    def analyze_json_file(self, file_path: str) -> Dict:
        """Analyze a JSON file containing scraped articles."""
        print(f"📄 Analyzing file: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                articles = json.load(f)
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            return {}
        
        if not isinstance(articles, list):
            print("❌ File does not contain a list of articles")
            return {}
        
        print(f"📊 Found {len(articles)} articles to analyze")
        
        total_chars_found = defaultdict(int)
        articles_with_issues = 0
        
        for i, article in enumerate(articles):
            article_id = f"article_{i}"
            
            # Check title
            title_chars = {}
            if 'title' in article and article['title']:
                title_chars = self.detect_invisible_chars_in_text(
                    article['title'], f"{article_id}_title"
                )
            
            # Check content
            content_chars = {}
            if 'content' in article and article['content']:
                content_chars = self.detect_invisible_chars_in_text(
                    article['content'], f"{article_id}_content"
                )
            
            # Combine results
            all_chars_in_article = {}
            for chars_dict in [title_chars, content_chars]:
                for char, count in chars_dict.items():
                    all_chars_in_article[char] = all_chars_in_article.get(char, 0) + count
                    total_chars_found[char] += count
            
            # Track articles with issues
            if all_chars_in_article:
                articles_with_issues += 1
                self.affected_articles[file_path].append({
                    'article_index': i,
                    'url': article.get('url', 'No URL'),
                    'title_preview': article.get('title', 'No title')[:50] + '...' if article.get('title') else 'No title',
                    'invisible_chars': all_chars_in_article
                })
        
        print(f"✅ Analysis complete: {articles_with_issues}/{len(articles)} articles have invisible Unicode issues")
        
        return {
            'file_path': file_path,
            'total_articles': len(articles),
            'affected_articles': articles_with_issues,
            'invisible_chars_found': dict(total_chars_found)
        }
    
    def generate_report(self, analysis_results: List[Dict]) -> str:
        """Generate a comprehensive report of findings."""
        report = []
        report.append("🔍 INVISIBLE UNICODE CHARACTER DETECTION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Summary
        total_files = len(analysis_results)
        total_articles = sum(r['total_articles'] for r in analysis_results)
        total_affected = sum(r['affected_articles'] for r in analysis_results)
        
        report.append(f"📊 SUMMARY")
        report.append(f"   Files analyzed: {total_files}")
        report.append(f"   Total articles: {total_articles}")
        report.append(f"   Articles with issues: {total_affected}")
        report.append(f"   Clean articles: {total_articles - total_affected}")
        report.append("")
        
        # Detailed findings per file
        for result in analysis_results:
            if result['invisible_chars_found']:
                report.append(f"📄 FILE: {result['file_path']}")
                report.append(f"   Articles: {result['total_articles']}")
                report.append(f"   Affected: {result['affected_articles']}")
                report.append("")
                
                # Sort by frequency
                sorted_chars = sorted(
                    result['invisible_chars_found'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                report.append("   🚨 INVISIBLE CHARACTERS FOUND:")
                for char_desc, count in sorted_chars:
                    report.append(f"      {char_desc}: {count:,} occurrences")
                
                report.append("")
        
        # Overall character frequency
        all_chars = defaultdict(int)
        for result in analysis_results:
            for char, count in result['invisible_chars_found'].items():
                all_chars[char] += count
        
        if all_chars:
            report.append("🌐 OVERALL CHARACTER FREQUENCY (across all files)")
            sorted_all_chars = sorted(all_chars.items(), key=lambda x: x[1], reverse=True)
            for char_desc, total_count in sorted_all_chars:
                report.append(f"   {char_desc}: {total_count:,} total occurrences")
            report.append("")
        
        # Recommendations
        report.append("💡 RECOMMENDATIONS")
        if all_chars:
            report.append("   ✅ Create cleaning script to remove these characters:")
            priority_chars = []
            for char_desc, count in sorted(all_chars.items(), key=lambda x: x[1], reverse=True):
                # Extract the actual character from description
                char = char_desc.split(' (')[0]
                priority_chars.append(f"'{char}'")
                
            report.append("   " + ", ".join(priority_chars[:10]))  # Show top 10
            
            report.append("")
            report.append("   🔧 Apply cleaning to:")
            for result in analysis_results:
                if result['affected_articles'] > 0:
                    report.append(f"      - {result['file_path']} ({result['affected_articles']} articles)")
        else:
            report.append("   ✅ No invisible Unicode characters found!")
            report.append("   📊 All analyzed text is clean.")
        
        return "\n".join(report)
    
    def get_cleaning_character_list(self) -> List[str]:
        """Get list of characters that need to be cleaned based on detection."""
        chars_to_clean = []
        
        # Get all detected characters across files
        for file_articles in self.affected_articles.values():
            for article_info in file_articles:
                for char_desc in article_info['invisible_chars'].keys():
                    # Extract the actual character from description
                    char = char_desc.split(' (')[0]
                    if char not in chars_to_clean:
                        chars_to_clean.append(char)
        
        return chars_to_clean


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 detect_invisible_unicode.py <file1.json> [file2.json] ...")
        print("Example: python3 detect_invisible_unicode.py data/raw_data/session_*/dvb_*.json")
        sys.exit(1)
    
    detector = InvisibleUnicodeDetector()
    analysis_results = []
    
    print("🔍 Starting invisible Unicode character detection...")
    print()
    
    for file_path in sys.argv[1:]:
        try:
            result = detector.analyze_json_file(file_path)
            if result:
                analysis_results.append(result)
        except Exception as e:
            print(f"❌ Error analyzing {file_path}: {e}")
    
    if not analysis_results:
        print("❌ No files were successfully analyzed.")
        return
    
    # Generate and display report
    print("\n" + "="*60)
    report = detector.generate_report(analysis_results)
    print(report)
    
    # Save report to file
    from datetime import datetime
    import os
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    report_file = f"logs/invisible_unicode_detection_report_{timestamp}.txt"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n💾 Report saved to: {report_file}")
    
    # Generate character list for cleaner script
    chars_to_clean = detector.get_cleaning_character_list()
    if chars_to_clean:
        chars_file = f"logs/characters_to_clean_{timestamp}.json"
        with open(chars_file, 'w', encoding='utf-8') as f:
            json.dump({
                'characters_to_clean': chars_to_clean,
                'character_details': detector.all_suspects,
                'detection_timestamp': timestamp,
                'files_analyzed': [r['file_path'] for r in analysis_results]
            }, f, ensure_ascii=False, indent=2)
        
        print(f"📋 Character list saved to: {chars_file}")
        print(f"   Use this file with the cleaner script to remove {len(chars_to_clean)} types of invisible characters")


if __name__ == "__main__":
    main()