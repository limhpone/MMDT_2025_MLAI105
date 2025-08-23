import json
import re
import unicodedata
import os
import shutil

class MyanmarJSONCleaner:
    def __init__(self):
        """Initialize Myanmar JSON cleaner for processing scraped articles"""
        pass
    
    def normalize_unicode(self, text):
        """Normalize Unicode characters"""
        return unicodedata.normalize('NFC', text)
    
    def clean_whitespace(self, text):
        """Clean and normalize whitespace"""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def remove_unwanted_chars(self, text):
        """Remove unwanted characters while preserving Myanmar text"""
        # Keep Myanmar chars, basic punctuation, numbers, and common symbols
        pattern = r'[^\u1000-\u109F\u0020-\u007E\u00A0-\u00FF\uAA60-\uAA7F\uA9E0-\uA9FF\u2000-\u206F\u2070-\u209F\u20A0-\u20CF]'
        text = re.sub(pattern, ' ', text)
        return text
    
    def remove_extra_punctuation(self, text):
        """Remove excessive punctuation while keeping essential ones"""
        text = re.sub(r'[,.!?;:]{2,}', '.', text)
        text = re.sub(r'\.{3,}', '...', text)
        return text
    
    def segment_sentences(self, text):
        """Segment text into sentences using Myanmar sentence ending patterns as delimiters"""
        if not text:
            return text
        
        # Split on Myanmar sentence endings
        sentences = re.split(r'(တယ်။|သည်။|ပြီ။|မည်။|မယ်။)', text)
        
        # Rejoin sentence endings with their sentences and add newlines
        result = []
        for i in range(0, len(sentences), 2):
            if i < len(sentences):
                sentence = sentences[i].strip()
                # Add the ending if it exists
                if i + 1 < len(sentences):
                    sentence += sentences[i + 1]
                if sentence:
                    result.append(sentence)
        
        return '\n'.join(result)
    
    def clean_text(self, text):
        """Clean and segment text"""
        if not text or not isinstance(text, str):
            return ""
        
        # Basic cleaning
        text = self.normalize_unicode(text)
        text = self.remove_unwanted_chars(text)
        text = self.clean_whitespace(text)
        text = self.remove_extra_punctuation(text)
        
        # Segment into sentences
        text = self.segment_sentences(text)
        
        return text
    
    def clean_article(self, article):
        """Clean a single article from JSON"""
        # Only keep title and content for training
        cleaned_article = {}
        
        # Clean title
        if 'title' in article:
            cleaned_article['title'] = self.clean_text(article['title'])
        
        # Clean content with sentence segmentation
        if 'content' in article:
            cleaned_article['content'] = self.clean_text(article['content'])
        
        # Add cleaned timestamp for tracking
        cleaned_article['cleaned_at'] = True
        
        return cleaned_article
    
    def clean_json_file(self, input_file, output_file):
        """Clean JSON file containing articles"""
        print(f"Cleaning JSON: {input_file}")
        
        # Load JSON data
        with open(input_file, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        
        if not isinstance(articles, list):
            raise ValueError("JSON file should contain a list of articles")
        
        cleaned_articles = []
        skipped_count = 0
        total_articles = len(articles)
        
        print(f"Processing {total_articles} articles...")
        
        for i, article in enumerate(articles):
            if not isinstance(article, dict):
                skipped_count += 1
                continue
                
            # Check if article has required fields
            if 'title' not in article or 'content' not in article:
                skipped_count += 1
                continue
            
            # Skip if title or content is empty
            if not article['title'].strip() or not article['content'].strip():
                skipped_count += 1
                continue
            
            # Clean the article
            cleaned_article = self.clean_article(article)
            cleaned_articles.append(cleaned_article)
            
            # Progress tracking
            if (i + 1) % 50 == 0 or (i + 1) == total_articles:
                progress = (i + 1) / total_articles * 100
                print(f"  Progress: {i + 1}/{total_articles} ({progress:.1f}%)")
        
        # Write cleaned JSON
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_articles, f, ensure_ascii=False, indent=2)
        
        print(f"Cleaned {len(cleaned_articles)} articles (skipped {skipped_count}) to: {output_file}")
        return len(cleaned_articles)

def main():
    """Clean all JSON files and output to to_process folder"""
    cleaner = MyanmarJSONCleaner()
    
    # Import project utilities
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from utils import get_data_directories
    
    # Get clean directory paths
    dirs = get_data_directories()
    input_dir = dirs['raw_to_process']
    output_dir = dirs['cleaned_to_process']
    done_dir = dirs['raw_done']
    
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(done_dir, exist_ok=True)
    
    # Process all JSON files in the to_process directory
    if not os.path.exists(input_dir):
        print(f"Input directory does not exist: {input_dir}")
        return
    
    # Get all .json files in the to_process directory
    files_to_process = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    
    if not files_to_process:
        print(f"No .json files found in: {input_dir}")
        return
    
    print(f"Found {len(files_to_process)} JSON files to process: {files_to_process}")
    
    total_articles = 0
    
    for filename in files_to_process:
        input_file = os.path.join(input_dir, filename)
        # Change extension to .json for output
        output_filename = filename if filename.endswith('.json') else filename + '.json'
        output_file = os.path.join(output_dir, output_filename)
        
        if os.path.exists(input_file):
            count = cleaner.clean_json_file(input_file, output_file)
            total_articles += count
            
            # Move processed file to done
            done_file = os.path.join(done_dir, filename)
            shutil.move(input_file, done_file)
            print(f"Moved {filename} to raw/done/")
    
    print(f"\nCleaning complete!")
    print(f"Total articles: {total_articles}")
    print(f"Files ready for preprocessing in: {output_dir}")

if __name__ == "__main__":
    main()