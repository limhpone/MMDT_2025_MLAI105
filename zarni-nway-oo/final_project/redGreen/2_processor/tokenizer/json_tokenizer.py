import json
import os
import sys
import shutil
import time
from datetime import datetime, timedelta

# Add myWord to path
sys.path.append(os.path.dirname(__file__))
from myWord import MyWord

class OptimizedMyWord:
    """Optimized Myanmar word tokenizer using original myWord but with performance improvements"""
    
    def __init__(self):
        """Initialize with performance optimizations while keeping myWord accuracy"""
        try:
            # Initialize dictionaries with optimizations
            script_dir = os.path.dirname(os.path.abspath(__file__))
            resources_dir = os.path.join(script_dir, 'myWord', 'resources')
            
            # Try to find dictionary files
            unigram_word_bin = os.path.join(resources_dir, 'unigram-word.bin')
            bigram_word_bin = os.path.join(resources_dir, 'combined_bigram', 'bigram-word.bin')
            
            # Check if dictionaries exist
            if os.path.exists(unigram_word_bin) and os.path.exists(bigram_word_bin):
                # Import word segmentation module
                sys.path.append(os.path.join(script_dir, 'myWord', 'helper'))
                import word_segment as wseg
                
                print("⚡ Loading optimized myWord dictionaries...")
                # Initialize global dictionaries with optimized cache
                wseg.P_unigram = wseg.ProbDist(unigram_word_bin, True)
                wseg.P_bigram = wseg.ProbDist(bigram_word_bin, False)
                
                self.wseg = wseg
                self.initialized = True
                print("✅ Optimized myWord tokenizer initialized")
            else:
                print(f"Warning: Dictionary files not found at {unigram_word_bin} or {bigram_word_bin}")
                self.initialized = False
                
        except Exception as e:
            print(f"Warning: Failed to initialize OptimizedMyWord: {e}")
            self.initialized = False
    
    def segment(self, text):
        """Segment Myanmar text into words using optimized myWord"""
        if not self.initialized:
            # Fallback to simple whitespace tokenization
            return text.strip().split() if text.strip() else []
        
        if not text.strip():
            return []
        
        try:
            # Use optimized Viterbi algorithm for word segmentation
            _, words = self.wseg.viterbi(text.strip())
            return words
        except Exception as e:
            print(f"Warning: Word segmentation failed: {e}")
            # Fallback to simple whitespace tokenization
            return text.strip().split() if text.strip() else []

class MyanmarJSONTokenizer:
    def __init__(self, use_optimized=True):
        """Initialize Myanmar JSON tokenizer using myWord with performance optimizations"""
        
        self.use_optimized = use_optimized
        
        # Initialize myWord tokenizer
        try:
            print("🔄 Initializing myWord tokenizer (this may take a moment)...", flush=True)
            start_init_time = time.time()
            
            if use_optimized:
                print("⚡ Using optimized myWord with larger cache...", flush=True)
                self.tokenizer = OptimizedMyWord()
            else:
                print("🐌 Using standard myWord (slower)...", flush=True)
                self.tokenizer = MyWord()
                
            init_time = time.time() - start_init_time
            print(f"✅ myWord tokenizer initialized successfully in {init_time:.2f}s", flush=True)
        except Exception as e:
            print(f"❌ Failed to initialize myWord: {e}", flush=True)
            print("⚠️  Using fallback tokenization", flush=True)
            self.tokenizer = None
    
    def create_progress_bar(self, current, total, width=50):
        """Create a visual progress bar"""
        percentage = (current / total) * 100
        filled = int(width * current // total)
        bar = '█' * filled + '░' * (width - filled)
        return f"[{bar}] {percentage:.1f}%"
    
    def format_time(self, seconds):
        """Format seconds into human readable time"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds // 60
            remaining_seconds = seconds % 60
            return f"{minutes:.0f}m {remaining_seconds:.0f}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours:.0f}h {minutes:.0f}m"
    
    def tokenize_text(self, text, max_chunk_size=15000):
        """Tokenize Myanmar text using myWord with chunking for long texts and aggressive error handling"""
        if not text or not isinstance(text, str):
            return ""
        
        # Skip extremely long texts that are likely to cause segfaults
        if len(text) > 40000:
            print(f"⚠️  Skipping extremely long text ({len(text)} chars) - potential segfault risk")
            return text  # Return as-is for extremely long texts
        
        try:
            if self.tokenizer and hasattr(self.tokenizer, 'initialized') and self.tokenizer.initialized:
                # For very long texts, process in smaller chunks to prevent memory issues
                if len(text) > max_chunk_size:
                    print(f"🔄 Processing long text ({len(text)} chars) in chunks...")
                    # Split by sentences/newlines first to preserve meaning
                    sentences = text.split('\n')
                    tokenized_sentences = []
                    current_chunk = ""
                    
                    for sentence in sentences:
                        # If adding this sentence would exceed chunk size, process current chunk
                        if len(current_chunk + sentence) > max_chunk_size and current_chunk:
                            try:
                                # Tokenize current chunk with timeout protection
                                chunk_tokenized = self.tokenizer.segment(current_chunk.strip())
                                if isinstance(chunk_tokenized, list):
                                    tokenized_sentences.extend(chunk_tokenized)
                                else:
                                    tokenized_sentences.append(chunk_tokenized)
                            except Exception as chunk_e:
                                print(f"⚠️  Chunk tokenization failed: {chunk_e}")
                                # Fallback: simple word splitting for this chunk
                                tokenized_sentences.extend(current_chunk.strip().split())
                            current_chunk = sentence
                        else:
                            current_chunk += ("\n" if current_chunk else "") + sentence
                    
                    # Process remaining chunk
                    if current_chunk.strip():
                        try:
                            chunk_tokenized = self.tokenizer.segment(current_chunk.strip())
                            if isinstance(chunk_tokenized, list):
                                tokenized_sentences.extend(chunk_tokenized)
                            else:
                                tokenized_sentences.append(chunk_tokenized)
                        except Exception as chunk_e:
                            print(f"⚠️  Final chunk tokenization failed: {chunk_e}")
                            tokenized_sentences.extend(current_chunk.strip().split())
                    
                    return ' '.join(tokenized_sentences) if tokenized_sentences else text
                else:
                    # Normal processing for texts under the limit
                    tokenized = self.tokenizer.segment(text)
                    if isinstance(tokenized, list):
                        return ' '.join(tokenized)
                    return tokenized
            else:
                # Fallback: basic space-based tokenization
                return text
        except (MemoryError, OSError) as e:
            print(f"⚠️  Memory/system error during tokenization: {e}")
            return text  # Return original text on memory errors
        except Exception as e:
            print(f"⚠️  Tokenization error: {e}")
            # Fallback: return original text
            return text
    
    def tokenize_article(self, article):
        """Tokenize a single article while maintaining JSON structure"""
        tokenized_article = article.copy()
        
        # Tokenize title
        if 'title' in article:
            tokenized_article['title'] = self.tokenize_text(article['title'])
        
        # Tokenize content (maintaining sentence structure with newlines)
        if 'content' in article:
            content = article['content']
            # Split by newlines to preserve sentence boundaries
            sentences = content.split('\n')
            tokenized_sentences = []
            
            for sentence in sentences:
                if sentence.strip():  # Skip empty lines
                    tokenized_sentence = self.tokenize_text(sentence.strip())
                    tokenized_sentences.append(tokenized_sentence)
            
            # Join back with newlines to preserve sentence structure
            tokenized_article['content'] = '\n'.join(tokenized_sentences)
        
        # Add tokenization metadata
        tokenized_article['tokenized_at'] = True
        
        return tokenized_article
    
    def tokenize_batch_parallel(self, articles_batch):
        """Tokenize a batch of articles - can be used with multiprocessing"""
        tokenized_batch = []
        failed_count = 0
        
        for article in articles_batch:
            if not isinstance(article, dict):
                failed_count += 1
                continue
                
            try:
                tokenized_article = self.tokenize_article(article)
                tokenized_batch.append(tokenized_article)
            except Exception as e:
                failed_count += 1
                continue
        
        return tokenized_batch, failed_count

    def tokenize_json_file(self, input_file, output_file, batch_size=100, use_parallel=False):
        """Tokenize JSON file with incremental saving and resume functionality"""
        filename = os.path.basename(input_file)
        print(f"\n🔤 Tokenizing: {filename}")
        print("=" * 60)
        
        # Check for existing partial output and resume point
        resume_file = output_file + '.resume'
        start_batch = 0
        existing_tokenized = []
        
        if os.path.exists(resume_file):
            try:
                with open(resume_file, 'r') as f:
                    resume_data = json.load(f)
                    start_batch = resume_data.get('last_completed_batch', 0)
                    print(f"🔄 Resuming from batch {start_batch + 1}")
                    
                # Load existing partial results if available
                if os.path.exists(output_file):
                    with open(output_file, 'r', encoding='utf-8') as f:
                        existing_tokenized = json.load(f)
                    print(f"📂 Loaded {len(existing_tokenized)} existing tokenized articles")
            except Exception as e:
                print(f"⚠️  Could not resume: {e}. Starting from beginning.")
                start_batch = 0
                existing_tokenized = []
        
        # Load JSON data
        with open(input_file, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        
        if not isinstance(articles, list):
            raise ValueError("JSON file should contain a list of articles")
        
        tokenized_articles = existing_tokenized.copy()
        failed_count = 0
        total_articles = len(articles)
        
        # Determine batch size based on file size and article count
        file_size_mb = os.path.getsize(input_file) / (1024 * 1024)
        if file_size_mb > 30:  # Large files
            batch_size = 25  # Smaller for better memory management
        elif file_size_mb > 15:  # Medium files  
            batch_size = 50
        else:  # Small files
            batch_size = 100
            
        print(f"📊 File size: {file_size_mb:.1f}MB, using batch size: {batch_size}")
        print(f"📊 Total articles to process: {total_articles}")
        print(f"📊 Starting from batch {start_batch + 1}")
        print(f"🚀 Starting incremental batch tokenization...")
        print()
        
        # Timing variables
        start_time = time.time()
        
        # Process articles in batches to prevent memory issues
        for batch_start in range(start_batch * batch_size, total_articles, batch_size):
            batch_end = min(batch_start + batch_size, total_articles)
            batch_articles = articles[batch_start:batch_end]
            batch_num = (batch_start // batch_size) + 1
            total_batches = (total_articles + batch_size - 1) // batch_size
            
            # Skip already processed batches
            if batch_num <= start_batch:
                continue
                
            print(f"🔄 Processing batch {batch_num}/{total_batches} (articles {batch_start+1}-{batch_end})...")
            
            batch_tokenized = []
            batch_failed = 0
            
            for i, article in enumerate(batch_articles):
                global_index = batch_start + i
                
                if not isinstance(article, dict):
                    batch_failed += 1
                    continue
                
                # Pre-check for problematic articles
                try:
                    content_len = len(article.get('content', ''))
                    title_len = len(article.get('title', ''))
                    total_len = content_len + title_len
                    
                    # Log long articles
                    if total_len > 20000:
                        print(f"\n⚠️  Article {global_index} is very long ({total_len} chars)")
                    
                    # Skip extremely problematic articles that consistently cause segfaults
                    if total_len > 45000:
                        print(f"\n⚠️  SKIPPING article {global_index} - too long ({total_len} chars) - likely to cause segfault")
                        batch_failed += 1
                        continue
                    
                    # Tokenize the article with extra error handling
                    tokenized_article = self.tokenize_article(article)
                    batch_tokenized.append(tokenized_article)
                        
                except MemoryError as e:
                    print(f"\n❌ Memory error on article {global_index}: {e}")
                    print(f"    Article length: {len(article.get('content', '')) + len(article.get('title', ''))}")
                    batch_failed += 1
                    # Force memory cleanup
                    import gc
                    gc.collect()
                    continue
                except Exception as e:
                    print(f"\n❌ Error tokenizing article {global_index}: {e}")
                    print(f"    Article length: {len(article.get('content', '')) + len(article.get('title', ''))}")
                    batch_failed += 1
                    continue
                
                # Show progress within batch for large batches
                if len(batch_articles) > 20 and (i + 1) % 10 == 0:
                    batch_progress = (i + 1) / len(batch_articles) * 100
                    print(f"   Batch progress: {i+1}/{len(batch_articles)} ({batch_progress:.0f}%)", flush=True)
            
            # Add batch results to main list (only new ones)
            tokenized_articles.extend(batch_tokenized)
            failed_count += batch_failed
            
            # CRITICAL: Save incrementally after each batch
            print(f"💾 Saving progress after batch {batch_num}...")
            try:
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(tokenized_articles, f, ensure_ascii=False, indent=2)
                
                # Save resume point
                resume_data = {
                    'last_completed_batch': batch_num,
                    'total_articles_processed': len(tokenized_articles),
                    'failed_count': failed_count,
                    'timestamp': time.time()
                }
                with open(resume_file, 'w') as f:
                    json.dump(resume_data, f, indent=2)
                    
                print(f"✅ Saved {len(tokenized_articles)} articles so far")
            except Exception as e:
                print(f"❌ Failed to save progress: {e}")
                # Continue processing but warn user
            
            # CRITICAL: Clear caches to prevent memory buildup
            if hasattr(self.tokenizer, 'wseg'):
                # Clear the LRU cache for viterbi function
                self.tokenizer.wseg.viterbi.cache_clear()
                print("🧹 Cleared tokenizer cache")
            
            # Force garbage collection
            import gc
            gc.collect()
            print(f"🧹 Memory cleanup completed")
            
            # Small delay to prevent overwhelming the system
            time.sleep(0.2)
            
            # Calculate and show overall progress
            current_time = time.time()
            current_count = batch_end
            elapsed_time = current_time - start_time
            avg_time_per_article = elapsed_time / current_count if current_count > 0 else 0
            remaining_articles = total_articles - current_count
            eta_seconds = remaining_articles * avg_time_per_article
            
            # Create progress display
            progress_bar = self.create_progress_bar(current_count, total_articles)
            elapsed_str = self.format_time(elapsed_time)
            eta_str = self.format_time(eta_seconds) if remaining_articles > 0 else "0s"
            
            # Articles per second
            articles_per_sec = current_count / elapsed_time if elapsed_time > 0 else 0
            
            # Show progress
            progress_line = (f"{progress_bar} "
                           f"{current_count}/{total_articles} | "
                           f"⚡ {articles_per_sec:.1f} art/s | "
                           f"⏱️  {elapsed_str} elapsed | "
                           f"🕓 ETA: {eta_str}")
            
            print(progress_line, flush=True)
            
            # Force garbage collection after each batch to free memory
            import gc
            gc.collect()
            
            # Small delay to prevent overwhelming the system
            time.sleep(0.1)
        
        # Final newline after progress
        print()
        
        # Final save (should already be saved, but just to be sure)
        print(f"\n💾 Final save of tokenized articles...")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        save_start_time = time.time()
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(tokenized_articles, f, ensure_ascii=False, indent=2)
        save_time = time.time() - save_start_time
        
        # Clean up resume file on successful completion
        if os.path.exists(resume_file):
            os.remove(resume_file)
            print(f"🧹 Removed resume file (processing completed)")
        
        # Final statistics
        total_time = time.time() - start_time
        avg_time = total_time / len(tokenized_articles) if tokenized_articles else 0
        
        print(f"✅ Tokenization completed!")
        print(f"📈 Statistics:")
        print(f"   • Successfully tokenized: {len(tokenized_articles)} articles")
        print(f"   • Failed: {failed_count} articles")
        print(f"   • Success rate: {len(tokenized_articles)/(len(tokenized_articles)+failed_count)*100:.1f}%")
        print(f"   • Processing time: {self.format_time(total_time)}")
        print(f"   • Average per article: {avg_time:.2f}s")
        print(f"   • Final save time: {save_time:.2f}s")
        print(f"   • Output: {output_file}")
        
        return len(tokenized_articles)

def main():
    """Tokenize all preprocessed JSON files with enhanced progress tracking"""
    print("\n" + "=" * 80)
    print("🔤 MYANMAR TEXT TOKENIZATION PIPELINE")
    print("=" * 80)
    
    # Use optimized myWord by default for better performance while maintaining accuracy
    tokenizer = MyanmarJSONTokenizer(use_optimized=True)
    
    # Use relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, '..', '..')
    input_dir = os.path.join(project_root, "data", "preprocessed", "to_process")
    output_dir = os.path.join(project_root, "data", "tokenized", "to_process")
    done_dir = os.path.join(project_root, "data", "preprocessed", "done")
    
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(done_dir, exist_ok=True)
    
    # Process all JSON files in the to_process directory
    if not os.path.exists(input_dir):
        print(f"❌ Input directory does not exist: {input_dir}")
        return
    
    files_to_process = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    
    if not files_to_process:
        print(f"⚠️  No .json files found in: {input_dir}")
        return
    
    print(f"📁 Found {len(files_to_process)} JSON files to tokenize")
    print(f"📂 Input directory: {input_dir}")
    print(f"📂 Output directory: {output_dir}")
    print()
    
    # Overall timing
    overall_start_time = time.time()
    total_articles = 0
    total_files_processed = 0
    
    for file_index, filename in enumerate(files_to_process, 1):
        print(f"\n📄 Processing file {file_index}/{len(files_to_process)}: {filename}")
        print("-" * 60)
        
        input_file = os.path.join(input_dir, filename)
        output_file = os.path.join(output_dir, filename)
        
        if os.path.exists(input_file):
            file_start_time = time.time()
            
            # Tokenize the file
            count = tokenizer.tokenize_json_file(input_file, output_file)
            
            file_time = time.time() - file_start_time
            total_articles += count
            total_files_processed += 1
            
            # Move processed file to done
            done_file = os.path.join(done_dir, filename)
            shutil.move(input_file, done_file)
            print(f"📦 Moved {filename} to preprocessed/done/")
            
            # Show file completion stats
            overall_elapsed = time.time() - overall_start_time
            files_remaining = len(files_to_process) - file_index
            avg_time_per_file = overall_elapsed / file_index
            eta_seconds = files_remaining * avg_time_per_file
            
            print(f"⏱️  File processed in: {tokenizer.format_time(file_time)}")
            if files_remaining > 0:
                print(f"🕓 Estimated time remaining: {tokenizer.format_time(eta_seconds)}")
        else:
            print(f"⚠️  File not found: {input_file}")
    
    # Final statistics
    total_time = time.time() - overall_start_time
    
    print("\n" + "=" * 80)
    print("🎉 TOKENIZATION PIPELINE COMPLETED!")
    print("=" * 80)
    print(f"📊 Final Statistics:")
    print(f"   • Files processed: {total_files_processed}/{len(files_to_process)}")
    print(f"   • Total articles tokenized: {total_articles}")
    print(f"   • Total processing time: {tokenizer.format_time(total_time)}")
    if total_articles > 0:
        print(f"   • Average time per article: {total_time/total_articles:.3f}s")
    if total_files_processed > 0:
        print(f"   • Average time per file: {tokenizer.format_time(total_time/total_files_processed)}")
    print(f"   • Articles per second: {total_articles/total_time:.2f}")
    print(f"📁 Tokenized files ready for labeling in: {output_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()