import re
import unicodedata
import os

class MyanmarTextCleaner:
    def __init__(self):
        """Initialize Myanmar text cleaner for basic cleaning only"""
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
        pattern = r'[^\u1000-\u109F\u0020-\u007E\u00A0-\u00FF\uAA60-\uAA7F\uA9E0-\uA9FF\u2000-\u206F\u2070-\u209F\u20A0-\u20CF]'
        text = re.sub(pattern, ' ', text)
        return text
    
    def remove_extra_punctuation(self, text):
        """Remove excessive punctuation while keeping essential ones"""
        text = re.sub(r'[,.!?;:]{2,}', '.', text)
        text = re.sub(r'\.{3,}', '...', text)
        return text
    
    def clean_text(self, text):
        """Basic text cleaning pipeline"""
        if not text or not isinstance(text, str):
            return ""
        
        text = self.normalize_unicode(text)
        text = self.remove_unwanted_chars(text)
        text = self.clean_whitespace(text)
        text = self.remove_extra_punctuation(text)
        text = self.clean_whitespace(text)
        
        return text
    
    def break_long_text(self, text, max_length=1500):
        """Break long text into smaller chunks at word boundaries"""
        if len(text) <= max_length:
            return text
        
        # Split text by spaces to get words
        words = text.split()
        chunks = []
        current_chunk = ""
        
        for word in words:
            # Check if adding this word would exceed max_length
            test_chunk = current_chunk + " " + word if current_chunk else word
            
            if len(test_chunk) <= max_length:
                current_chunk = test_chunk
            else:
                # Start new chunk
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = word
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        # Join chunks with newline to keep them separate for tokenization
        return '\n'.join(chunks)
    
    def detect_article_boundaries(self, text):
        """Detect article boundaries in unstructured text"""
        # Common Myanmar article boundary patterns
        patterns = [
            r'\n\n+',  # Double newlines
            r'\n\s*[០-៩\u1040-\u1049]+[.)]\s*',  # Numbered sections
            r'\n\s*[က-ဿ]+\s*[၀-၉\u1040-\u1049]*\s*[-–—]\s*',  # Location-date patterns
            r'\n\s*[က-ဿ]{2,}\s*[၀-၉\u1040-\u1049]{1,2}\s*\n',  # Date patterns
        ]
        
        # Try patterns in order of preference
        for pattern in patterns:
            splits = re.split(pattern, text)
            if len(splits) > 1:
                return [chunk.strip() for chunk in splits if chunk.strip()]
        
        # If no patterns found, split by estimated article length
        return self.split_by_estimated_length(text)
    
    def split_by_estimated_length(self, text, target_length=2000):
        """Split text by estimated article length at sentence boundaries"""
        sentences = re.split(r'[။!?]\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(test_chunk) <= target_length:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk + "။")  # Add Myanmar period
                current_chunk = sentence
        
        if current_chunk:
            if not current_chunk.endswith('။'):
                current_chunk += "။"
            chunks.append(current_chunk)
        
        return chunks
    
    def extract_title_from_content(self, content):
        """Extract potential title from article content"""
        lines = content.split('\n')
        first_line = lines[0].strip() if lines else ""
        
        # If first line is short and doesn't end with period, likely a title
        if len(first_line) <= 100 and not first_line.endswith(('။', '.', '!', '?')):
            title = first_line
            remaining_content = '\n'.join(lines[1:]).strip() if len(lines) > 1 else ""
            return title, remaining_content
        
        # Otherwise, generate title from first few words
        words = content.split()[:8]  # First 8 words as title
        title = ' '.join(words)
        return title, content
    
    def clean_file_to_txt(self, input_file, output_file):
        """Clean input file and output in same txt format - handles both structured and unstructured data"""
        print(f"Cleaning: {input_file}")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        cleaned_articles = []
        
        # First try structured format (title/content pairs)
        blocks = re.split(r'\n\s*\n+', content.strip())
        structured_articles = []
        
        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) >= 2:
                title = lines[0].strip()
                article_content = ' '.join(lines[1:]).strip()
                
                # Check if this looks like a title (short, no period)
                if len(title) <= 150 and not title.endswith(('။', '.', '!', '?')):
                    structured_articles.append((title, article_content))
        
        # If we found structured articles, use them
        if structured_articles:
            print(f"Found {len(structured_articles)} structured articles")
            for title, article_content in structured_articles:
                cleaned_title = self.clean_text(title)
                cleaned_content = self.clean_text(article_content)
                cleaned_content = self.break_long_text(cleaned_content, max_length=500)
                
                if cleaned_title and cleaned_content:
                    cleaned_articles.append(f"{cleaned_title}\n{cleaned_content}")
        
        # Otherwise, treat as unstructured continuous text
        else:
            print("No structured format detected, processing as continuous text")
            # Clean the entire content first
            cleaned_content = self.clean_text(content)
            
            # Detect article boundaries
            article_chunks = self.detect_article_boundaries(cleaned_content)
            print(f"Detected {len(article_chunks)} article chunks")
            
            for chunk in article_chunks:
                if len(chunk.strip()) < 50:  # Skip very short chunks
                    continue
                
                # Extract title and content
                title, article_content = self.extract_title_from_content(chunk)
                
                # Break long content into manageable chunks
                article_content = self.break_long_text(article_content, max_length=500)
                
                if title and article_content:
                    cleaned_articles.append(f"{title}\n{article_content}")
        
        # Write cleaned articles
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(cleaned_articles))
        
        print(f"Cleaned {len(cleaned_articles)} articles to: {output_file}")
        return len(cleaned_articles)

def main():
    """Clean all input files and output to to_process folder"""
    import shutil
    
    cleaner = MyanmarTextCleaner()
    
    # Use relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, '..', '..')
    input_dir = os.path.join(project_root, "data", "raw", "to_process")
    output_dir = os.path.join(project_root, "data", "cleaned", "to_process")
    done_dir = os.path.join(project_root, "data", "raw", "done")
    
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(done_dir, exist_ok=True)
    
    # Process all files in the to_process directory
    if not os.path.exists(input_dir):
        print(f"Input directory does not exist: {input_dir}")
        return
    
    # Get all .txt files in the to_process directory
    files_to_process = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    
    if not files_to_process:
        print(f"No .txt files found in: {input_dir}")
        return
    
    print(f"Found {len(files_to_process)} files to process: {files_to_process}")
    
    total_articles = 0
    
    for filename in files_to_process:
        input_file = os.path.join(input_dir, filename)
        output_file = os.path.join(output_dir, filename)  # Keep same filename
        
        if os.path.exists(input_file):
            count = cleaner.clean_file_to_txt(input_file, output_file)
            total_articles += count
            
            # Move processed file to done folder
            done_file = os.path.join(done_dir, filename)
            shutil.move(input_file, done_file)
            print(f"Moved {filename} to raw/done/")
        else:
            print(f"File not found: {input_file}")
    
    print(f"\nCleaning complete!")
    print(f"Total articles: {total_articles}")
    print(f"Files ready for tokenization in: {output_dir}")

if __name__ == "__main__":
    main()