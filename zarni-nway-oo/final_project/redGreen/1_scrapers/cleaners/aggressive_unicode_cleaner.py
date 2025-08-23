#!/usr/bin/env python3
"""
Aggressive Unicode Character Cleaner
This script will completely remove ALL invisible Unicode characters
"""

import json
import sys
import unicodedata
from datetime import datetime

def aggressive_clean_text(text):
    """Aggressively clean text by removing ALL invisible Unicode characters."""
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
        category = unicodedata.category(char)
        
        # Keep Myanmar script characters (U+1000-U+109F, U+AA60-U+AA7F)
        is_myanmar = (0x1000 <= ord_c <= 0x109F) or (0xAA60 <= ord_c <= 0xAA7F)
        
        # Keep ASCII characters (0-127)
        is_ascii = ord_c < 127
        
        # Keep common safe characters
        is_safe = ord_c in [0x09, 0x0A, 0x0D]  # Tab, newline, carriage return
        
        # Only keep characters that are safe
        if is_ascii or is_myanmar or is_safe:
            cleaned_chars.append(char)
        else:
            # For debugging: log what we're removing
            # print(f"Removing: {repr(char)} U+{ord_c:04X} - {unicodedata.name(char, 'UNNAMED')}")
            pass
    
    text = ''.join(cleaned_chars)
    
    # Step 4: Normalize whitespace
    import re
    text = re.sub(r' +', ' ', text)  # Multiple spaces → single space
    text = re.sub(r'\n\s*\n+', '\n\n', text)  # Multiple newlines → double newline
    
    return text.strip()

def clean_json_file(input_file, output_file):
    """Clean a JSON file containing articles."""
    print(f"🧹 Aggressively cleaning: {input_file}")
    
    # Load data
    with open(input_file, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    
    print(f"📊 Processing {len(articles)} articles...")
    
    # Clean each article
    cleaned_articles = []
    articles_changed = 0
    
    for i, article in enumerate(articles):
        cleaned_article = article.copy()
        
        # Clean title
        if 'title' in article and article['title']:
            original_title = article['title']
            cleaned_title = aggressive_clean_text(original_title)
            cleaned_article['title'] = cleaned_title
            if original_title != cleaned_title:
                articles_changed += 1
        
        # Clean content
        if 'content' in article and article['content']:
            original_content = article['content']
            cleaned_content = aggressive_clean_text(original_content)
            cleaned_article['content'] = cleaned_content
            if original_content != cleaned_content:
                articles_changed += 1
        
        cleaned_articles.append(cleaned_article)
        
        if (i + 1) % 1000 == 0:
            print(f"   Processed {i + 1}/{len(articles)} articles...")
    
    # Save cleaned data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_articles, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Aggressive cleaning complete!")
    print(f"   📄 Output: {output_file}")
    print(f"   📊 Articles modified: {articles_changed}")
    
    return cleaned_articles

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 aggressive_unicode_cleaner.py <input_file.json> <output_file.json>")
        print("Example: python3 aggressive_unicode_cleaner.py data.json cleaned_data.json")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    print("🚀 Starting AGGRESSIVE Unicode cleaning...")
    print("   This will remove ALL invisible characters completely")
    print()
    
    cleaned_articles = clean_json_file(input_file, output_file)
    
    # Verify cleaning
    print("\n🔍 Verifying cleaning...")
    with open(output_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    found_invisible = []
    for char in content:
        ord_c = ord(char)
        category = unicodedata.category(char)
        
        if (category in ['Cf', 'Cc'] and ord_c > 31) or ord_c in [0x00A0, 0x200B, 0x200C, 0x200D, 0x2060, 0x2068, 0x2069]:
            char_name = unicodedata.name(char, f'UNNAMED-{ord_c:04X}')
            key = f'{repr(char)} U+{ord_c:04X} - {char_name}'
            if key not in found_invisible:
                found_invisible.append(key)
    
    # Save verification report
    import os
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    verification_report = []
    verification_report.append("🧹 AGGRESSIVE UNICODE CLEANER - VERIFICATION REPORT")
    verification_report.append("=" * 60)
    verification_report.append("")
    verification_report.append(f"Input file: {input_file}")
    verification_report.append(f"Output file: {output_file}")
    verification_report.append(f"Processing time: {timestamp}")
    verification_report.append(f"Articles processed: {len(cleaned_articles)}")
    verification_report.append("")
    
    if found_invisible:
        verification_report.append("❌ REMAINING INVISIBLE CHARACTERS DETECTED:")
        for char in found_invisible:
            verification_report.append(f"   {char}")
        verification_report.append("")
        verification_report.append("⚠️  WARNING: File may still have invisible character issues")
        
        print("❌ Still found invisible characters:")
        for char in found_invisible[:10]:  # Show first 10
            print(f"   {char}")
    else:
        verification_report.append("✅ SUCCESS: NO INVISIBLE CHARACTERS FOUND")
        verification_report.append("   File is completely clean and ready for use.")
        verification_report.append("   VSCode warnings should be eliminated.")
        
        print("✅ SUCCESS: No invisible characters found!")
        print("   File is completely clean and ready to use.")
    
    # Save verification report
    report_file = f"logs/aggressive_cleaning_verification_{timestamp}.txt"
    try:
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(verification_report))
        print(f"\n💾 Verification report saved to: {report_file}")
    except Exception as e:
        print(f"⚠️  Could not save verification report: {e}")

if __name__ == "__main__":
    main()