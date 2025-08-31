import re
import os
import hashlib
from collections import Counter

class MyanmarTextPreprocessor:
    def __init__(self, min_length=30, max_length=2000, duplicate_threshold=0.8):
        """
        Initialize Myanmar text preprocessor for quality validation
        
        Args:
            min_length: Minimum text length to keep
            max_length: Maximum text length to keep  
            duplicate_threshold: Similarity threshold for duplicate detection
        """
        self.min_length = min_length
        self.max_length = max_length
        self.duplicate_threshold = duplicate_threshold
        self.seen_hashes = set()
        self.processed_texts = []
    
    def is_myanmar_text(self, text):
        """Check if text contains Myanmar characters"""
        myanmar_char_pattern = r'[\u1000-\u109F]'
        myanmar_chars = re.findall(myanmar_char_pattern, text)
        return len(myanmar_chars) > 10  # Need at least 10 Myanmar characters
    
    def check_text_quality(self, text):
        """Check various quality metrics for text"""
        issues = []
        
        # Length check
        if len(text) < self.min_length:
            issues.append(f"too_short ({len(text)} chars)")
        if len(text) > self.max_length:
            issues.append(f"too_long ({len(text)} chars)")
        
        # Myanmar content check
        if not self.is_myanmar_text(text):
            issues.append("no_myanmar_content")
        
        # Check for excessive whitespace
        whitespace_ratio = len(re.findall(r'\s', text)) / len(text) if text else 0
        if whitespace_ratio > 0.3:
            issues.append(f"excessive_whitespace ({whitespace_ratio:.2f})")
        
        # Check for repeated patterns
        words = text.split()
        if len(words) > 0:
            word_counts = Counter(words)
            most_common_ratio = word_counts.most_common(1)[0][1] / len(words)
            if most_common_ratio > 0.3:
                issues.append(f"repetitive_content ({most_common_ratio:.2f})")
        
        return issues
    
    def get_text_hash(self, text):
        """Get hash of text for duplicate detection"""
        # Normalize text for comparison
        normalized = re.sub(r'\s+', ' ', text.lower().strip())
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()
    
    def is_duplicate(self, text):
        """Check if text is similar to previously seen texts"""
        text_hash = self.get_text_hash(text)
        
        if text_hash in self.seen_hashes:
            return True
        
        # Check similarity with existing texts (simple approach)
        words = set(text.lower().split())
        for existing_text in self.processed_texts[-50:]:  # Check last 50 texts only
            existing_words = set(existing_text.lower().split())
            if words and existing_words:
                similarity = len(words & existing_words) / len(words | existing_words)
                if similarity > self.duplicate_threshold:
                    return True
        
        self.seen_hashes.add(text_hash)
        self.processed_texts.append(text)
        return False
    
    def validate_title_content_pair(self, title, content):
        """Validate title and content separately"""
        title_issues = self.check_text_quality(title) if title else ["missing_title"]
        content_issues = self.check_text_quality(content) if content else ["missing_content"]
        
        # Title-specific checks
        if title and len(title) > 200:
            title_issues.append("title_too_long")
        
        # Content-specific checks  
        if content and title and title.lower() == content.lower():
            content_issues.append("title_content_identical")
        
        return title_issues, content_issues
    
    def preprocess_file(self, input_file, output_file):
        """Preprocess cleaned file and apply quality filters"""
        print(f"Preprocessing: {input_file}")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split articles (title\ncontent format)
        articles = content.strip().split('\n\n')
        
        valid_articles = []
        stats = {
            'total': len(articles),
            'valid': 0,
            'too_short': 0,
            'too_long': 0,
            'no_myanmar': 0,
            'duplicates': 0,
            'quality_issues': 0
        }
        
        for article in articles:
            lines = article.strip().split('\n', 1)
            if len(lines) != 2:
                continue
                
            title, content = lines
            title = title.strip()
            content = content.strip()
            
            # Validate title and content
            title_issues, content_issues = self.validate_title_content_pair(title, content)
            all_issues = title_issues + content_issues
            
            # Check for duplicates
            full_text = f"{title} {content}"
            if self.is_duplicate(full_text):
                stats['duplicates'] += 1
                continue
            
            # Categorize rejection reasons
            rejected = False
            for issue in all_issues:
                if 'too_short' in issue:
                    stats['too_short'] += 1
                    rejected = True
                elif 'too_long' in issue:
                    stats['too_long'] += 1  
                    rejected = True
                elif 'no_myanmar_content' in issue:
                    stats['no_myanmar'] += 1
                    rejected = True
                else:
                    stats['quality_issues'] += 1
            
            # Keep article if no serious issues
            serious_issues = ['too_short', 'too_long', 'no_myanmar_content', 'missing_title', 'missing_content']
            has_serious_issues = any(any(serious in issue for serious in serious_issues) for issue in all_issues)
            
            if not has_serious_issues and not rejected:
                valid_articles.append(article)
                stats['valid'] += 1
        
        # Write valid articles
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(valid_articles))
        
        # Print statistics
        print(f"Preprocessing Stats:")
        print(f"  Total articles: {stats['total']}")
        print(f"  Valid articles: {stats['valid']}")
        print(f"  Rejected - Too short: {stats['too_short']}")
        print(f"  Rejected - Too long: {stats['too_long']}")
        print(f"  Rejected - No Myanmar: {stats['no_myanmar']}")
        print(f"  Rejected - Duplicates: {stats['duplicates']}")
        print(f"  Rejected - Quality issues: {stats['quality_issues']}")
        print(f"  Processed file saved to: {output_file}")
        
        return stats

def main():
    """Preprocess all cleaned files and output to to_process folder"""
    import shutil
    
    preprocessor = MyanmarTextPreprocessor()
    
    # Use relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, '..', '..')
    input_dir = os.path.join(project_root, "data", "cleaned", "to_process")
    output_dir = os.path.join(project_root, "data", "preprocessed", "to_process")
    done_dir = os.path.join(project_root, "data", "cleaned", "done")
    
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(done_dir, exist_ok=True)
    
    # Process all files in the to_process directory
    if not os.path.exists(input_dir):
        print(f"Input directory does not exist: {input_dir}")
        return
    
    files_to_process = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    
    if not files_to_process:
        print(f"No .txt files found in: {input_dir}")
        return
    
    print(f"Found {len(files_to_process)} files to preprocess: {files_to_process}")
    
    total_stats = {'total': 0, 'valid': 0, 'too_short': 0, 'too_long': 0, 'no_myanmar': 0, 'duplicates': 0, 'quality_issues': 0}
    
    for filename in files_to_process:
        input_file = os.path.join(input_dir, filename)
        output_file = os.path.join(output_dir, filename)
        
        if os.path.exists(input_file):
            stats = preprocessor.preprocess_file(input_file, output_file)
            
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
    print(f"    Too short: {total_stats['too_short']}")
    print(f"    Too long: {total_stats['too_long']}")
    print(f"    No Myanmar: {total_stats['no_myanmar']}")
    print(f"    Duplicates: {total_stats['duplicates']}")
    print(f"    Quality issues: {total_stats['quality_issues']}")
    print(f"\nFiles ready for tokenization in: {output_dir}")

if __name__ == "__main__":
    main()