import json
import re
import os
import shutil
import hashlib
from collections import Counter

class MyanmarJSONPreprocessor:
    def __init__(self, min_length=5, max_length=20000, duplicate_threshold=0.9):
        """
        Initialize Myanmar JSON preprocessor with relaxed constraints for low-resource language
        
        Args:
            min_length: Minimum text length (very permissive)
            max_length: Maximum text length (very permissive) 
            duplicate_threshold: Similarity threshold for duplicates (CURRENTLY UNUSED - duplicate detection disabled)
        """
        self.min_length = min_length
        self.max_length = max_length
        self.duplicate_threshold = duplicate_threshold
        self.seen_hashes = set()
        self.processed_articles = []
    
    def is_myanmar_text(self, text):
        """Check if text contains Myanmar characters - very lenient"""
        myanmar_char_pattern = r'[\u1000-\u109F]'
        myanmar_chars = re.findall(myanmar_char_pattern, text)
        # Only need 3 Myanmar characters minimum
        return len(myanmar_chars) >= 3
    
    def check_text_quality(self, text):
        """Check basic quality metrics - very permissive for low-resource language"""
        issues = []
        
        if not text:
            issues.append("empty_text")
            return issues
        
        # Very lenient length checks
        if len(text) < self.min_length:
            issues.append(f"too_short ({len(text)} chars)")
        if len(text) > self.max_length:
            issues.append(f"too_long ({len(text)} chars)")
        
        # Myanmar content check - but very lenient
        if not self.is_myanmar_text(text):
            issues.append("minimal_myanmar_content")
        
        # Only check for extremely excessive whitespace
        whitespace_ratio = len(re.findall(r'\s', text)) / len(text) if text else 0
        if whitespace_ratio > 0.8:  # Much more permissive
            issues.append(f"excessive_whitespace ({whitespace_ratio:.2f})")
        
        return issues
    
    def get_article_hash(self, article):
        """Get hash of article for duplicate detection"""
        # Use title + content for hashing
        text = f"{article.get('title', '')} {article.get('content', '')}"
        normalized = re.sub(r'\s+', ' ', text.lower().strip())
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()
    
    def is_duplicate(self, article):
        """Check if article is duplicate - high threshold for Myanmar language"""
        # TEMPORARILY DISABLED: Uncomment below to re-enable duplicate detection
        return False
        
        # Original duplicate detection logic (commented out for training with all data):
        # article_hash = self.get_article_hash(article)
        # 
        # if article_hash in self.seen_hashes:
        #     return True
        # 
        # # Check similarity with existing articles (simple approach)
        # title_words = set(article.get('title', '').lower().split())
        # content_words = set(article.get('content', '').lower().split())
        # current_words = title_words | content_words
        # 
        # # Only check against recent articles to avoid O(n²) complexity
        # for existing_article in self.processed_articles[-20:]:  # Check last 20 only
        #     existing_title = set(existing_article.get('title', '').lower().split())
        #     existing_content = set(existing_article.get('content', '').lower().split())
        #     existing_words = existing_title | existing_content
        #     
        #     if current_words and existing_words:
        #         similarity = len(current_words & existing_words) / len(current_words | existing_words)
        #         if similarity > self.duplicate_threshold:
        #             return True
        # 
        # self.seen_hashes.add(article_hash)
        # self.processed_articles.append(article)
        # return False
    
    def validate_article(self, article):
        """Validate article - very permissive for low-resource language"""
        issues = []
        
        if not isinstance(article, dict):
            return ["invalid_format"]
        
        # Check required fields
        if 'title' not in article:
            issues.append("missing_title")
        if 'content' not in article:
            issues.append("missing_content")
        
        # Check content quality - but very lenient
        if 'title' in article:
            title_issues = self.check_text_quality(article['title'])
            # Only care about serious title issues
            serious_title_issues = [i for i in title_issues if 'empty' in i]
            issues.extend([f"title_{i}" for i in serious_title_issues])
        
        if 'content' in article:
            content_issues = self.check_text_quality(article['content'])
            # Only care about serious content issues  
            serious_content_issues = [i for i in content_issues if 'empty' in i or 'too_long' in i]
            issues.extend([f"content_{i}" for i in serious_content_issues])
        
        return issues
    
    def preprocess_json_file(self, input_file, output_file):
        """Preprocess JSON file with minimal filtering"""
        print(f"Preprocessing JSON: {input_file}")
        
        # Load JSON data
        with open(input_file, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        
        if not isinstance(articles, list):
            raise ValueError("JSON file should contain a list of articles")
        
        valid_articles = []
        stats = {
            'total': len(articles),
            'valid': 0,
            'invalid_format': 0,
            'missing_fields': 0,
            'duplicates': 0,
            'serious_issues': 0
        }
        
        total_articles = len(articles)
        print(f"Processing {total_articles} articles...")
        
        for i, article in enumerate(articles):
            # Validate article
            issues = self.validate_article(article)
            
            # Check for duplicates
            if self.is_duplicate(article):
                stats['duplicates'] += 1
                continue
            
            # Categorize issues
            serious_issues = ['invalid_format', 'missing_title', 'missing_content', 'empty_text']
            has_serious_issues = any(any(serious in issue for serious in serious_issues) for issue in issues)
            
            if has_serious_issues:
                if any('invalid_format' in issue for issue in issues):
                    stats['invalid_format'] += 1
                elif any(field in issue for issue in issues for field in ['missing_title', 'missing_content']):
                    stats['missing_fields'] += 1
                else:
                    stats['serious_issues'] += 1
            else:
                # Keep article - very permissive for low-resource language
                # Add preprocessing metadata
                article['preprocessed_at'] = True
                valid_articles.append(article)
                stats['valid'] += 1
            
            # Progress tracking
            if (i + 1) % 50 == 0 or (i + 1) == total_articles:
                progress = (i + 1) / total_articles * 100
                print(f"  Progress: {i + 1}/{total_articles} ({progress:.1f}%)")
        
        # Write valid articles
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(valid_articles, f, ensure_ascii=False, indent=2)
        
        # Print statistics
        print(f"Preprocessing Stats:")
        print(f"  Total articles: {stats['total']}")
        print(f"  Valid articles: {stats['valid']} ({stats['valid']/stats['total']*100:.1f}%)")
        print(f"  Rejected - Invalid format: {stats['invalid_format']}")
        print(f"  Rejected - Missing fields: {stats['missing_fields']}")
        print(f"  Rejected - Duplicates: {stats['duplicates']}")
        print(f"  Rejected - Serious issues: {stats['serious_issues']}")
        print(f"  Processed file saved to: {output_file}")
        
        return stats

def main():
    """Preprocess all cleaned JSON files"""
    preprocessor = MyanmarJSONPreprocessor()
    
    # Import project utilities
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from utils import get_data_directories
    
    # Get clean directory paths
    dirs = get_data_directories()
    input_dir = dirs['cleaned_to_process']
    output_dir = dirs['preprocessed_to_process']
    done_dir = dirs['cleaned_done']
    
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(done_dir, exist_ok=True)
    
    # Process all JSON files in the to_process directory
    if not os.path.exists(input_dir):
        print(f"Input directory does not exist: {input_dir}")
        return
    
    files_to_process = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    
    if not files_to_process:
        print(f"No .json files found in: {input_dir}")
        return
    
    print(f"Found {len(files_to_process)} JSON files to preprocess: {files_to_process}")
    
    total_stats = {'total': 0, 'valid': 0, 'invalid_format': 0, 'missing_fields': 0, 'duplicates': 0, 'serious_issues': 0}
    
    for filename in files_to_process:
        input_file = os.path.join(input_dir, filename)
        output_file = os.path.join(output_dir, filename)
        
        if os.path.exists(input_file):
            stats = preprocessor.preprocess_json_file(input_file, output_file)
            
            # Accumulate stats
            for key in total_stats:
                total_stats[key] += stats[key]
            
            # Move processed file to done
            done_file = os.path.join(done_dir, filename)
            shutil.move(input_file, done_file)
            print(f"Moved {filename} to cleaned/done/")
    
    print(f"\nOverall Preprocessing Complete!")
    print(f"Total Stats:")
    print(f"  Total articles: {total_stats['total']}")
    print(f"  Valid articles: {total_stats['valid']} ({total_stats['valid']/total_stats['total']*100:.1f}%)")
    print(f"  Rejection breakdown:")
    print(f"    Invalid format: {total_stats['invalid_format']}")
    print(f"    Missing fields: {total_stats['missing_fields']}")
    print(f"    Duplicates: {total_stats['duplicates']}")
    print(f"    Serious issues: {total_stats['serious_issues']}")
    print(f"\nFiles ready for tokenization in: {output_dir}")

if __name__ == "__main__":
    main()