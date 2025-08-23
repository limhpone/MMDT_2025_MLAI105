#!/usr/bin/env python3
"""
Deep Unicode Character Inspector
Byte-level analysis to find any invisible/problematic characters that VSCode might detect
"""

import json
import sys
import unicodedata
from typing import Dict, List, Tuple


class DeepUnicodeInspector:
    def __init__(self):
        self.suspicious_ranges = [
            # Control characters
            (0x0000, 0x001F, "C0 Control Characters"),
            (0x007F, 0x009F, "C1 Control Characters"),
            
            # Invisible/formatting characters
            (0x2000, 0x206F, "General Punctuation (includes invisible chars)"),
            (0x2070, 0x209F, "Superscripts and Subscripts"),
            (0x20A0, 0x20CF, "Currency Symbols"),
            (0x2100, 0x214F, "Letterlike Symbols"),
            (0xFE00, 0xFE0F, "Variation Selectors"),
            (0xFE20, 0xFE2F, "Combining Half Marks"),
            (0xFEFF, 0xFEFF, "Byte Order Mark"),
            
            # Arabic/Hebrew formatting
            (0x0600, 0x06FF, "Arabic"),
            (0x0750, 0x077F, "Arabic Supplement"),
            
            # Other potentially problematic ranges
            (0x1100, 0x11FF, "Hangul Jamo"),
            (0x3130, 0x318F, "Hangul Compatibility Jamo"),
            (0xAC00, 0xD7AF, "Hangul Syllables"),
        ]
    
    def analyze_character(self, char: str) -> Dict:
        """Analyze a single character in detail."""
        ord_c = ord(char)
        category = unicodedata.category(char)
        
        try:
            name = unicodedata.name(char)
        except ValueError:
            name = f"UNNAMED-{ord_c:04X}"
        
        # Check if in Myanmar ranges
        is_myanmar = (0x1000 <= ord_c <= 0x109F) or (0xAA60 <= ord_c <= 0xAA7F)
        
        # Check various properties
        is_printable = char.isprintable()
        is_space = char.isspace()
        is_control = category.startswith('C')
        is_format = category == 'Cf'
        is_combining = category.startswith('M')
        
        # Check if in suspicious range
        suspicious_range = None
        for start, end, desc in self.suspicious_ranges:
            if start <= ord_c <= end and not is_myanmar:
                suspicious_range = desc
                break
        
        return {
            'char': char,
            'ord': ord_c,
            'hex': f'U+{ord_c:04X}',
            'category': category,
            'name': name,
            'is_myanmar': is_myanmar,
            'is_printable': is_printable,
            'is_space': is_space,
            'is_control': is_control,
            'is_format': is_format,
            'is_combining': is_combining,
            'suspicious_range': suspicious_range,
            'bytes_utf8': char.encode('utf-8'),
            'repr': repr(char)
        }
    
    def inspect_text_sample(self, text: str, sample_size: int = 1000) -> Dict:
        """Inspect a sample of text for detailed character analysis."""
        sample = text[:sample_size]
        char_analysis = []
        suspicious_chars = {}
        
        for i, char in enumerate(sample):
            analysis = self.analyze_character(char)
            
            # Flag suspicious characters
            if (not analysis['is_myanmar'] and 
                (analysis['is_control'] or 
                 analysis['is_format'] or 
                 analysis['suspicious_range'] or
                 (analysis['is_combining'] and analysis['ord'] > 127) or
                 (analysis['is_space'] and analysis['ord'] != 0x20 and 
                  analysis['ord'] != 0x09 and analysis['ord'] != 0x0A and analysis['ord'] != 0x0D))):
                
                key = f"{analysis['repr']} {analysis['hex']} - {analysis['name']}"
                if key not in suspicious_chars:
                    suspicious_chars[key] = []
                suspicious_chars[key].append({
                    'position': i,
                    'context': sample[max(0, i-5):i+6],
                    'analysis': analysis
                })
        
        return {
            'sample_length': len(sample),
            'total_chars': len(sample),
            'suspicious_chars': suspicious_chars,
            'sample_text_repr': repr(sample[:200]) + ('...' if len(sample) > 200 else '')
        }
    
    def inspect_json_file(self, file_path: str) -> Dict:
        """Inspect a JSON file for invisible Unicode characters."""
        print(f"🔍 Deep inspecting file: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                articles = json.load(f)
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            return {}
        
        if not isinstance(articles, list):
            print("❌ File does not contain a list of articles")
            return {}
        
        print(f"📊 Inspecting {len(articles)} articles...")
        
        all_suspicious = {}
        articles_with_issues = []
        
        # Sample first few articles for detailed inspection
        for i, article in enumerate(articles[:10]):  # Check first 10 articles in detail
            article_issues = {}
            
            # Check title
            if 'title' in article and article['title']:
                title_inspection = self.inspect_text_sample(article['title'], 500)
                if title_inspection['suspicious_chars']:
                    article_issues['title'] = title_inspection
            
            # Check content sample
            if 'content' in article and article['content']:
                content_inspection = self.inspect_text_sample(article['content'], 1000)
                if content_inspection['suspicious_chars']:
                    article_issues['content'] = content_inspection
            
            if article_issues:
                articles_with_issues.append({
                    'article_index': i,
                    'url': article.get('url', 'No URL'),
                    'issues': article_issues
                })
                
                # Merge into overall suspicious chars
                for field, inspection in article_issues.items():
                    for char_key, occurrences in inspection['suspicious_chars'].items():
                        if char_key not in all_suspicious:
                            all_suspicious[char_key] = []
                        all_suspicious[char_key].extend(occurrences)
        
        # Quick scan of remaining articles for character frequency
        print(f"📈 Quick scanning remaining {len(articles) - 10} articles...")
        for i, article in enumerate(articles[10:], 10):
            if i % 500 == 0:
                print(f"   Scanned {i}/{len(articles)} articles...")
            
            # Quick check for non-printable/suspicious chars in title and content
            for field in ['title', 'content']:
                if field in article and article[field]:
                    text = article[field]
                    for char in text:
                        analysis = self.analyze_character(char)
                        if (not analysis['is_myanmar'] and 
                            (analysis['is_control'] or analysis['is_format'] or 
                             analysis['suspicious_range'])):
                            key = f"{analysis['repr']} {analysis['hex']} - {analysis['name']}"
                            if key not in all_suspicious:
                                all_suspicious[key] = []
                            all_suspicious[key].append({'article_index': i, 'field': field})
        
        return {
            'file_path': file_path,
            'total_articles': len(articles),
            'detailed_articles_checked': min(10, len(articles)),
            'articles_with_issues': articles_with_issues,
            'all_suspicious_chars': all_suspicious,
            'summary': {
                'total_suspicious_char_types': len(all_suspicious),
                'articles_with_detailed_issues': len(articles_with_issues)
            }
        }
    
    def generate_report(self, inspection_result: Dict) -> str:
        """Generate detailed inspection report."""
        if not inspection_result:
            return "❌ No inspection results available"
        
        report = []
        report.append("🔬 DEEP UNICODE CHARACTER INSPECTION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Summary
        report.append("📊 INSPECTION SUMMARY")
        report.append(f"   File: {inspection_result['file_path']}")
        report.append(f"   Total articles: {inspection_result['total_articles']:,}")
        report.append(f"   Detailed inspection: {inspection_result['detailed_articles_checked']} articles")
        report.append(f"   Articles with issues: {inspection_result['summary']['articles_with_detailed_issues']}")
        report.append(f"   Suspicious character types: {inspection_result['summary']['total_suspicious_char_types']}")
        report.append("")
        
        # Suspicious characters found
        if inspection_result['all_suspicious_chars']:
            report.append("🚨 SUSPICIOUS CHARACTERS DETECTED")
            for char_key, occurrences in inspection_result['all_suspicious_chars'].items():
                report.append(f"   {char_key}")
                report.append(f"      Occurrences: {len(occurrences)}")
                
                # Show some example contexts
                examples = occurrences[:3]  # Show first 3 examples
                for ex in examples:
                    if 'context' in ex:
                        report.append(f"      Context: {repr(ex['context'])}")
                    else:
                        report.append(f"      Article {ex.get('article_index', '?')}, Field: {ex.get('field', '?')}")
                
                if len(occurrences) > 3:
                    report.append(f"      ... and {len(occurrences) - 3} more occurrences")
                report.append("")
        
        # Detailed article issues
        if inspection_result['articles_with_issues']:
            report.append("📄 DETAILED ARTICLE ISSUES")
            for article_info in inspection_result['articles_with_issues']:
                report.append(f"   Article {article_info['article_index']}: {article_info['url']}")
                
                for field, inspection in article_info['issues'].items():
                    report.append(f"      {field.upper()}:")
                    report.append(f"         Sample text: {inspection['sample_text_repr']}")
                    for char_key, char_occurrences in inspection['suspicious_chars'].items():
                        report.append(f"         Suspicious: {char_key} ({len(char_occurrences)} times)")
                report.append("")
        
        # Recommendations
        report.append("💡 RECOMMENDATIONS")
        if inspection_result['all_suspicious_chars']:
            report.append("   🔧 Add these characters to the cleaning script:")
            chars_to_add = []
            for char_key in inspection_result['all_suspicious_chars'].keys():
                # Extract hex code
                if 'U+' in char_key:
                    hex_code = char_key.split('U+')[1].split(' ')[0]
                    chars_to_add.append(f"'\\u{hex_code.lower()}'")
            
            if chars_to_add:
                report.append("   " + ", ".join(chars_to_add[:10]))  # Show first 10
        else:
            report.append("   ✅ No suspicious characters detected in deep inspection!")
        
        return "\n".join(report)


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 deep_unicode_inspector.py <file.json>")
        print("Example: python3 deep_unicode_inspector.py data/cleaned_file.json")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    print("🔬 Starting deep Unicode character inspection...")
    print("   This will do byte-level analysis to find any hidden characters")
    print()
    
    inspector = DeepUnicodeInspector()
    result = inspector.inspect_json_file(file_path)
    
    if result:
        report = inspector.generate_report(result)
        print("\n" + "="*60)
        print(report)
        
        # Save report
        from datetime import datetime
        import os
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ensure logs directory exists
        os.makedirs("logs", exist_ok=True)
        
        report_file = f"logs/deep_unicode_inspection_{timestamp}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n💾 Detailed inspection report saved to: {report_file}")
    else:
        print("❌ Inspection failed")


if __name__ == "__main__":
    main()