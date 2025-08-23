#!/usr/bin/env python3
"""
Myanmar News Classification Bi-LSTM Pipeline
Main script for running the complete data processing and Bi-LSTM training pipeline
"""

import os
import sys
import time
import subprocess
import shutil
import re
import unicodedata
from datetime import datetime
from utils import get_data_directories

class BiLSTMPipeline:
    def __init__(self):
        """Initialize pipeline with script paths using utils"""
        from utils import find_project_root
        
        self.project_root = find_project_root()
        
        # Script paths (using centralized paths)
        self.scripts = {
            'scraper': os.path.join(self.project_root, '1_scrapers', 'advanced_scraper.py'),
            'cleaner': os.path.join(self.project_root, '2_processor', 'cleaner', 'json_cleaner.py'),
            'preprocessor': os.path.join(self.project_root, '2_processor', 'preprocessor', 'json_preprocessor.py'),
            'tokenizer': os.path.join(self.project_root, '2_processor', 'tokenizer', 'json_tokenizer.py'),
            'labeller': os.path.join(self.project_root, '2_processor', 'labeller', 'json_labeller.py'),
            'trainer': os.path.join(self.project_root, '3_trainer', 'trainer', 'bilstm_trainer.py'),
            'analyzer': os.path.join(self.project_root, '4_analyzer', 'article_analyzer.py')
        }
        
        # Initialize myword tokenizer
        self.myword_tokenizer = None
        self._initialize_myword_tokenizer()
        
        # Verify all scripts exist
        self.verify_scripts()
    
    def _initialize_myword_tokenizer(self):
        """Initialize myWord tokenizer for article processing"""
        try:
            # Add myword path to sys.path (clean structure)
            myword_path = os.path.join(self.project_root, '2_processor', 'tokenizer', 'myWord')
            if myword_path not in sys.path:
                sys.path.append(myword_path)
            
            from myword import MyWord
            self.myword_tokenizer = MyWord()
            
            if hasattr(self.myword_tokenizer, 'initialized') and self.myword_tokenizer.initialized:
                print("✅ MyWord tokenizer initialized successfully")
            else:
                print("⚠️  MyWord tokenizer initialized but may have issues")
                
        except Exception as e:
            print(f"❌ Failed to initialize MyWord tokenizer: {e}")
            print("⚠️  Article tokenization will use fallback method")
            self.myword_tokenizer = None
    
    def _clean_text(self, text):
        """Clean text similar to training pipeline"""
        # Normalize Unicode
        text = unicodedata.normalize('NFC', text)
        
        # Remove unwanted characters while preserving Myanmar text
        pattern = r'[^\u1000-\u109F\u0020-\u007E\u00A0-\u00FF\uAA60-\uAA7F\uA9E0-\uA9FF\u2000-\u206F\u2070-\u209F\u20A0-\u20CF]'
        text = re.sub(pattern, ' ', text)
        
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Remove excessive punctuation
        text = re.sub(r'[,.!?;:]{2,}', '.', text)
        text = re.sub(r'\.{3,}', '...', text)
        
        return text
    
    def _tokenize_text(self, text):
        """Tokenize text using myWord"""
        if not self.myword_tokenizer:
            print("⚠️  Using fallback tokenization (whitespace split)")
            return text.split()
        
        try:
            tokens = self.myword_tokenizer.segment(text)
            return tokens
        except Exception as e:
            print(f"❌ Error in myWord tokenization: {e}")
            return text.split()
    
    def _tokenize_raw_files(self, raw_dir, processed_dir, raw_files):
        """Tokenize raw files and save to processed directory (preserving line structure)"""
        for filename in raw_files:
            src_path = os.path.join(raw_dir, filename)
            dst_path = os.path.join(processed_dir, filename)
            
            print(f"   🔤 Tokenizing: {filename}")
            
            # Read raw file
            with open(src_path, 'r', encoding='utf-8') as f:
                raw_content = f.read().strip()
            
            # Tokenize while preserving line structure (like training pipeline)
            lines = raw_content.split('\n')
            tokenized_lines = []
            total_tokens = 0
            
            for line in lines:
                if line.strip():  # Skip empty lines but preserve structure
                    cleaned_line = self._clean_text(line.strip())
                    tokens = self._tokenize_text(cleaned_line)
                    tokenized_line = ' '.join(tokens)
                    tokenized_lines.append(tokenized_line)
                    total_tokens += len(tokens)
                else:
                    tokenized_lines.append('')  # Preserve empty lines
            
            # Join back with newlines to preserve structure
            tokenized_content = '\n'.join(tokenized_lines)
            
            # Save tokenized version
            with open(dst_path, 'w', encoding='utf-8') as f:
                f.write(tokenized_content)
            
            print(f"      Original length: {len(raw_content)} chars")
            print(f"      Tokenized length: {len(tokenized_content)} chars")
            print(f"      Total tokens: {total_tokens}")
            print(f"      Lines preserved: {len(lines)} → {len(tokenized_lines)}")
            print(f"      Sample: {tokenized_content[:100]}...")
    
    def verify_scripts(self):
        """Verify all required scripts exist"""
        missing_scripts = []
        for name, path in self.scripts.items():
            if not os.path.exists(path):
                missing_scripts.append(f"{name}: {path}")
        
        if missing_scripts:
            print("❌ Missing required scripts:")
            for script in missing_scripts:
                print(f"   {script}")
            sys.exit(1)
        
        print("✅ All pipeline scripts found")
    
    def print_progress(self, step, total_steps, description):
        """Print progress with visual indicator"""
        progress = "█" * int(20 * step / total_steps) + "░" * (20 - int(20 * step / total_steps))
        percentage = int(100 * step / total_steps)
        print(f"\n[{progress}] {percentage}% - {description}")
        print(f"Step {step}/{total_steps}: {description}")
        print("-" * 50)
    
    def run_script(self, script_name, description, check_data=True):
        """Run a script and handle errors with real-time output"""
        script_path = self.scripts[script_name]
        
        print(f"🚀 Running {description}...")
        print(f"Script: {script_path}")
        print("-" * 60)
        
        start_time = time.time()
        
        try:
            # Run the script with real-time output - no buffering
            import os
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            
            process = subprocess.Popen([sys.executable, '-u', script_path], 
                                     stdout=subprocess.PIPE, 
                                     stderr=subprocess.STDOUT,
                                     text=True, 
                                     bufsize=0,  # No buffering
                                     universal_newlines=True,
                                     env=env)
            
            output_lines = []
            
            # Stream output in real-time
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    line = output.strip()
                    print(line, flush=True)
                    output_lines.append(line)
            
            # Wait for process to complete
            return_code = process.poll()
            elapsed_time = time.time() - start_time
            
            # Check if no data was found (for pipeline failure)
            no_data_indicators = [
                "No .json files found",
                "Input directory does not exist",
                "No data to save"
            ]
            
            has_no_data = any(indicator in line for line in output_lines for indicator in no_data_indicators)
            
            if return_code == 0:
                print("-" * 60)
                if has_no_data and check_data:
                    print(f"⚠️  {description} completed but found no data to process!")
                    print(f"⏱️  Execution time: {elapsed_time:.2f} seconds")
                    return "no_data"
                else:
                    print(f"✅ {description} completed successfully!")
                    print(f"⏱️  Execution time: {elapsed_time:.2f} seconds")
                    return True
            else:
                print("-" * 60)
                print(f"❌ {description} failed with return code {return_code}!")
                print(f"⏱️  Execution time: {elapsed_time:.2f} seconds")
                return False
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            print("-" * 60)
            print(f"❌ {description} failed!")
            print(f"⏱️  Execution time: {elapsed_time:.2f} seconds")
            print(f"Error: {e}")
            return False
    
    def run_scraping(self):
        """Run the scraping process and point output to correct directory"""
        print("🔍 Starting Data Scraping")
        print("=" * 60)
        
        # Get directories using utils
        dirs = get_data_directories()
        os.makedirs(dirs['raw_scraped'], exist_ok=True)
        os.makedirs(dirs['raw_to_process'], exist_ok=True)
        
        print(f"📁 Output will be saved to: {dirs['raw_scraped']}")
        print(f"📁 Files will be moved to: {dirs['raw_to_process']}")
        print()
        print("Available scrapers:")
        print("  1. DVB News")
        print("  2. Myawady News") 
        print("  3. Khitthit News")
        print("  4. All sources (runs all three)")
        print()
        
        try:
            scraper_choice = input("Select scraper (1-4): ").strip()
            if scraper_choice not in ['1', '2', '3', '4']:
                print("❌ Invalid choice!")
                return False
                
            # Get number of articles
            while True:
                try:
                    num_articles = int(input("Number of articles to scrape (e.g., 100): "))
                    if num_articles > 0:
                        break
                    else:
                        print("Please enter a positive number")
                except ValueError:
                    print("Please enter a valid number")
            
            self.print_progress(1, 1, "Web Scraping")
            
            # Handle "All sources" option
            if scraper_choice == '4':
                sources = ['dvb', 'myawady', 'khitthit']
                articles_per_source = max(1, num_articles // 3)
                
                all_success = True
                for source in sources:
                    print(f"\n🔍 Scraping from {source.upper()}...")
                    success = self._run_single_scraper_simple(source, articles_per_source, dirs)
                    if not success:
                        all_success = False
                        print(f"⚠️  {source} scraping failed, continuing...")
                
                result = all_success
            else:
                # Single source
                source_map = {'1': 'dvb', '2': 'myawady', '3': 'khitthit'}
                source = source_map[scraper_choice]
                result = self._run_single_scraper_simple(source, num_articles, dirs)
            
            if result:
                print(f"\n📊 Scraping completed! Check: {dirs['raw_to_process']}")
            else:
                print("\n💥 Scraping failed!")
                return False
                
        except KeyboardInterrupt:
            print("\n⚠️  Scraping interrupted by user")
            return False
        except Exception as e:
            print(f"\n❌ Error during scraping: {e}")
            return False
        
        return True
    
    def _run_single_scraper_simple(self, source, num_articles, dirs):
        """Run scraper and let it work as designed"""
        # Arguments for the advanced scraper - let it use default behavior
        scraper_args = [
            '--source', source, 
            '--articles', str(num_articles)
            # No --output, let scraper use its default session directory
        ]
        
        # Run the scraper
        result = self.run_scraper_with_args(scraper_args)
        
        if result:
            # First move to scraped directory, then to to_process
            self._find_and_move_from_session(source, dirs['raw_scraped'], dirs['raw_to_process'])
        
        return result
    
    def _find_and_move_from_session(self, source, scraped_dir, to_process_dir):
        """Find article files from scraper's session directory and move to our structure"""
        import glob
        
        # Look for the scraper's session directories in the correct location
        session_pattern = os.path.join(scraped_dir, 'session_*')
        session_dirs = sorted(glob.glob(session_pattern), key=os.path.getmtime, reverse=True)
        
        if not session_dirs:
            print(f"   ❌ No session directories found for {source}")
            return False
        
        # Look in the latest session directory
        latest_session = session_dirs[0]
        
        # Find the main article JSON file (using the actual naming pattern from log)
        json_pattern = os.path.join(latest_session, f"{source}_raw_myanmar_*.json")
        json_files = glob.glob(json_pattern)
        
        for json_file in json_files:
            filename = os.path.basename(json_file)
            
            # Skip files that are not the main article file
            skip_patterns = ['_stats_', '_session_', '_training_', '_readable_', '_log_', '_debug_']
            if any(pattern in filename.lower() for pattern in skip_patterns):
                continue
            
            # Move the article file to to_process (ready for pipeline)
            dest_path = os.path.join(to_process_dir, filename)
            shutil.move(json_file, dest_path)
            print(f"   ✅ Article file ready: {filename}")
            return True
        
        print(f"   ❌ No main article file found for {source} in {latest_session}")
        return False
    
    def run_scraper_with_args(self, args):
        """Run the scraper script with specific arguments"""
        script_path = self.scripts['scraper']
        
        print(f"🚀 Running Web Scraper...")
        print(f"Script: {script_path}")
        print(f"Arguments: {' '.join(args)}")
        print("-" * 60)
        
        start_time = time.time()
        current_dir = os.getcwd()
        
        try:
            # Change to scrapers directory so scraper creates files there
            scrapers_dir = os.path.dirname(script_path)
            os.chdir(scrapers_dir)
            
            # Prepare environment
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            
            # Run scraper with arguments (run from scrapers directory)
            cmd = [sys.executable, '-u', os.path.basename(script_path)] + args
            process = subprocess.Popen(cmd, 
                                     stdout=subprocess.PIPE, 
                                     stderr=subprocess.STDOUT,
                                     text=True, 
                                     bufsize=0,
                                     universal_newlines=True,
                                     env=env)
            
            output_lines = []
            
            # Stream output in real-time
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    line = output.strip()
                    print(line, flush=True)
                    output_lines.append(line)
            
            return_code = process.poll()
            elapsed_time = time.time() - start_time
            
            if return_code == 0:
                print("-" * 60)
                print(f"✅ Web Scraping completed successfully!")
                print(f"⏱️  Execution time: {elapsed_time:.2f} seconds")
                return True
            else:
                print("-" * 60)
                print(f"❌ Web Scraping failed with return code {return_code}!")
                print(f"⏱️  Execution time: {elapsed_time:.2f} seconds")
                return False
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            print("-" * 60)
            print(f"❌ Web Scraping failed!")
            print(f"⏱️  Execution time: {elapsed_time:.2f} seconds")
            print(f"Error: {e}")
            return False
        finally:
            # Always change back to original directory
            os.chdir(current_dir)
    
    def clean_only(self):
        """Run only the cleaning step"""
        print("🧹 Starting Data Cleaning Only")
        print("=" * 60)
        
        self.print_progress(1, 1, "Data Cleaning")
        result = self.run_script('cleaner', 'JSON Cleaning')
        
        if result == "no_data":
            print("\n⚠️  No data found to clean!")
            return False
        elif result:
            print("\n🎉 Data cleaning completed successfully!")
        else:
            print("\n💥 Data cleaning failed!")
            return False
        
        return True
    
    def preprocess_only(self):
        """Run only the preprocessing step"""
        print("🔄 Starting Data Preprocessing Only")
        print("=" * 60)
        
        self.print_progress(1, 1, "Data Preprocessing")
        result = self.run_script('preprocessor', 'JSON Preprocessing')
        
        if result == "no_data":
            print("\n⚠️  No data found to preprocess!")
            return False
        elif result:
            print("\n🎉 Data preprocessing completed successfully!")
        else:
            print("\n💥 Data preprocessing failed!")
            return False
        
        return True
    
    def tokenize_only(self):
        """Run only the tokenization step"""
        print("🔤 Starting Data Tokenization Only")
        print("=" * 60)
        
        self.print_progress(1, 1, "Data Tokenization")
        result = self.run_script('tokenizer', 'JSON Tokenization')
        
        if result == "no_data":
            print("\n⚠️  No data found to tokenize!")
            return False
        elif result:
            print("\n🎉 Data tokenization completed successfully!")
        else:
            print("\n💥 Data tokenization failed!")
            return False
        
        return True
    
    def label_only(self):
        """Run only the labelling step"""
        print("🏷️  Starting Data Labelling Only")
        print("=" * 60)
        
        self.print_progress(1, 1, "Data Labelling")
        result = self.run_script('labeller', 'JSON Labelling')
        
        if result == "no_data":
            print("\n⚠️  No data found to label!")
            return False
        elif result:
            print("\n🎉 Data labelling completed successfully!")
        else:
            print("\n💥 Data labelling failed!")
            return False
        
        return True
    
    def detect_resume_step(self):
        """Detect which step to resume from based on existing data"""
        data_root = os.path.join(self.project_root, 'data')
        
        # Check for files in different stages
        raw_to_process = os.path.join(data_root, 'raw', 'to_process')
        cleaned_to_process = os.path.join(data_root, 'cleaned', 'to_process') 
        preprocessed_to_process = os.path.join(data_root, 'preprocessed', 'to_process')
        tokenized_to_process = os.path.join(data_root, 'tokenized', 'to_process')
        
        # Check for resume files (tokenization in progress)
        if os.path.exists(preprocessed_to_process):
            json_files = [f for f in os.listdir(preprocessed_to_process) if f.endswith('.json')]
            resume_files = [f for f in os.listdir(preprocessed_to_process) if f.endswith('.json.resume')]
            
            if resume_files:
                print(f"🔄 Found {len(resume_files)} resume files - tokenization was interrupted")
                return 3  # Resume tokenization
            elif json_files:
                print(f"📂 Found {len(json_files)} preprocessed files ready for tokenization")
                return 3  # Start tokenization
        
        # Check other stages
        if os.path.exists(tokenized_to_process):
            json_files = [f for f in os.listdir(tokenized_to_process) if f.endswith('.json')]
            if json_files:
                print(f"📂 Found {len(json_files)} tokenized files ready for labelling")
                return 4  # Start labelling
        
        if os.path.exists(cleaned_to_process):
            json_files = [f for f in os.listdir(cleaned_to_process) if f.endswith('.json')]
            if json_files:
                print(f"📂 Found {len(json_files)} cleaned files ready for preprocessing")
                return 2  # Start preprocessing
        
        if os.path.exists(raw_to_process):
            json_files = [f for f in os.listdir(raw_to_process) if f.endswith('.json')]
            if json_files:
                print(f"📂 Found {len(json_files)} raw files ready for cleaning")
                return 1  # Start cleaning
        
        print("⚠️  No data found in any processing stage")
        return 0

    def prepare_data(self):
        """Run the complete data preparation pipeline with smart resume"""
        print("📋 Starting Complete Data Preparation Pipeline")
        print("=" * 60)
        
        # Detect where to start/resume from
        start_step = self.detect_resume_step()
        
        if start_step == 0:
            print("💡 No data found to process. Please check your input data.")
            return False
        
        total_steps = 4
        steps = [
            ('cleaner', 'JSON Cleaning'),
            ('preprocessor', 'JSON Preprocessing'), 
            ('tokenizer', 'JSON Tokenization'),
            ('labeller', 'JSON Labelling')
        ]
        
        print(f"🚀 Resuming from step {start_step}: {steps[start_step-1][1]}")
        start_time = time.time()
        
        for i, (script_name, description) in enumerate(steps[start_step-1:], start_step):
            self.print_progress(i, total_steps, description)
            
            result = self.run_script(script_name, description)
            
            if result == "no_data":
                print(f"\n⚠️  Data preparation stopped at step {i}: {description}")
                print("💡 No data found to process. Please check your input data.")
                return False
            elif not result:
                print(f"\n💥 Data preparation failed at step {i}: {description}")
                return False
            
            print(f"✅ Step {i} completed successfully!")
        
        total_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("🎉 Complete Data Preparation Pipeline Finished!")
        print(f"⏱️  Total execution time: {total_time:.2f} seconds")
        print("📊 Data is now ready for training!")
        
        return True
    
    def train_model(self):
        """Run model training"""
        print("🤖 Starting Model Training")
        print("=" * 60)
        
        self.print_progress(1, 1, "Bi-LSTM Model Training")
        result = self.run_script('trainer', 'Bi-LSTM Training', check_data=False)
        
        if result:
            print("\n🎉 Model training completed successfully!")
        else:
            print("\n💥 Model training failed!")
            return False
        
        return True
    
    def analyze_articles(self):
        """Run article analysis with file processing workflow (raw → processed → done)"""
        print("📊 Starting Article Analysis Pipeline")
        print("=" * 60)
        
        # Define directories
        raw_dir = os.path.join(self.project_root, "data", "model_tester", "raw")
        processed_dir = os.path.join(self.project_root, "data", "model_tester", "processed")
        done_dir = os.path.join(self.project_root, "data", "model_tester", "done")
        
        # Create directories if they don't exist
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)
        os.makedirs(done_dir, exist_ok=True)
        
        print(f"📁 Raw directory created/verified: {raw_dir}")
        print(f"📁 Processed directory created/verified: {processed_dir}")
        print(f"📁 Done directory created/verified: {done_dir}")
        
        # Check if raw files exist
        if not os.path.exists(raw_dir):
            print(f"❌ Raw directory not found: {raw_dir}")
            return False
        
        raw_files = [f for f in os.listdir(raw_dir) if f.endswith('.txt')]
        if not raw_files:
            print(f"⚠️  No .txt files found in {raw_dir}")
            return False
        
        print(f"📂 Found {len(raw_files)} files to process")
        print(f"   Raw: {raw_dir}")
        print(f"   Processed: {processed_dir}")
        print(f"   Done: {done_dir}")
        print()
        
        start_time = time.time()
        
        try:
            # Step 1: Tokenize and save files to processed directory
            print("🔄 Step 1: Tokenizing and saving files to processed directory...")
            self._tokenize_raw_files(raw_dir, processed_dir, raw_files)
            
            # Step 2: Run analyzer on processed directory
            print("\n🤖 Step 2: Running article analysis...")
            self.print_progress(1, 1, "Article Analysis")
            
            # Temporarily change the analyzer script to use processed directory
            result = self.run_analyzer_on_processed(processed_dir)
            
            if not result:
                print("❌ Analysis failed!")
                return False
            
            # Step 3: Move files from processed to done
            print("\n📦 Step 3: Moving processed files to done directory...")
            for filename in raw_files:
                src = os.path.join(processed_dir, filename)
                dst = os.path.join(done_dir, filename)
                if os.path.exists(src):
                    shutil.move(src, dst)
                    print(f"   ✅ Moved: {filename}")
            
            total_time = time.time() - start_time
            
            print("\n" + "=" * 60)
            print("🎉 Article Analysis Pipeline Completed!")
            print(f"⏱️  Total execution time: {total_time:.2f} seconds")
            print(f"📊 Processed {len(raw_files)} articles")
            print(f"📁 Files moved to: {done_dir}")
            print("📈 Analysis reports generated in: 4_analyzer/analysis_outputs/")
            
            return True
            
        except Exception as e:
            print(f"❌ Error in analysis pipeline: {e}")
            return False
    
    def run_analyzer_on_processed(self, processed_dir):
        """Run the analyzer script on the processed directory"""
        script_path = self.scripts['analyzer']
        
        print(f"🚀 Running Article Analysis...")
        print(f"Script: {script_path}")
        print(f"Processing: {processed_dir}")
        print("-" * 60)
        
        start_time = time.time()
        
        try:
            # Prepare environment
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            
            # Set environment variable to tell analyzer to use processed directory
            env['PROCESSED_DIR'] = processed_dir
            
            # Run the analyzer script directly
            process = subprocess.Popen([sys.executable, '-u', script_path], 
                                     stdout=subprocess.PIPE, 
                                     stderr=subprocess.STDOUT,
                                     text=True, 
                                     bufsize=0,
                                     universal_newlines=True,
                                     env=env)
            
            output_lines = []
            
            # Stream output in real-time
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    line = output.strip()
                    print(line, flush=True)
                    output_lines.append(line)
            

            
            return_code = process.poll()
            elapsed_time = time.time() - start_time
            
            if return_code == 0:
                print("-" * 60)
                print(f"✅ Article Analysis completed successfully!")
                print(f"⏱️  Execution time: {elapsed_time:.2f} seconds")
                return True
            else:
                print("-" * 60)
                print(f"❌ Article Analysis failed with return code {return_code}!")
                print(f"⏱️  Execution time: {elapsed_time:.2f} seconds")
                return False
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            print("-" * 60)
            print(f"❌ Article Analysis failed!")
            print(f"⏱️  Execution time: {elapsed_time:.2f} seconds")
            print(f"Error: {e}")
            return False
    
    def create_temp_analyzer_script(self, processed_dir):
        """Create a temporary analyzer script that uses the processed directory"""
        temp_script_path = os.path.join(self.project_root, 'temp_analyzer.py')
        
        # Read the original analyzer script
        with open(self.scripts['analyzer'], 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        # Fix the myword import path for the temporary script location
        # Replace the relative path with an absolute path
        myword_path = os.path.join(self.project_root, '2_processor', 'tokenizer', 'myWord')
        
        modified_content = original_content.replace(
            "sys.path.append(os.path.join(os.path.dirname(__file__), '..', '1_processors', 'tokenizer', 'myWord'))",
            f'sys.path.append("{myword_path}")'
        )
        
        # Also handle the clean structure path
        modified_content = modified_content.replace(
            "sys.path.append(os.path.join(os.path.dirname(__file__), '..', '2_processor', 'tokenizer', 'myWord'))",
            f'sys.path.append("{myword_path}")'
        )
        
        # Modify the test_dir path to use processed directory
        modified_content = modified_content.replace(
            'test_dir = os.path.join(project_root, "data", "model_tester")',
            f'test_dir = "{processed_dir}"'
        )
        
        # Write the modified script
        with open(temp_script_path, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        
        return temp_script_path
    
    def cleanup_done_folders(self):
        """Clean up all done/ folders except raw/done"""
        print("🧹 Starting Cleanup of Done Folders")
        print("=" * 60)
        
        # Define folders to clean (excluding raw/done)
        folders_to_clean = [
            os.path.join(self.project_root, "data", "cleaned", "done"),
            os.path.join(self.project_root, "data", "preprocessed", "done"), 
            os.path.join(self.project_root, "data", "tokenized", "done"),
            os.path.join(self.project_root, "data", "labelled", "done")
        ]
        
        total_files_removed = 0
        folders_cleaned = 0
        
        for folder_path in folders_to_clean:
            folder_name = os.path.relpath(folder_path, self.project_root)
            
            if not os.path.exists(folder_path):
                print(f"⚠️  Folder doesn't exist: {folder_name}")
                continue
            
            # Get list of files in the folder
            files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
            
            if not files:
                print(f"✅ Already clean: {folder_name} (0 files)")
                continue
            
            print(f"🗑️  Cleaning: {folder_name} ({len(files)} files)")
            
            # Remove each file
            files_removed = 0
            for filename in files:
                file_path = os.path.join(folder_path, filename)
                try:
                    os.remove(file_path)
                    files_removed += 1
                except Exception as e:
                    print(f"   ❌ Error removing {filename}: {e}")
            
            if files_removed > 0:
                print(f"   ✅ Removed {files_removed} files from {folder_name}")
                total_files_removed += files_removed
                folders_cleaned += 1
        
        print("\n" + "=" * 60)
        print("🎉 Cleanup Complete!")
        print(f"📊 Summary:")
        print(f"   • Folders processed: {len(folders_to_clean)}")
        print(f"   • Folders cleaned: {folders_cleaned}")
        print(f"   • Total files removed: {total_files_removed}")
        print(f"   • Raw data preserved: data/raw/done/ (untouched)")
        print("=" * 60)
        
        return True
    
    def show_menu(self):
        """Display the main menu"""
        print("\n" + "=" * 60)
        print("🇲🇲 Myanmar News Classification Bi-LSTM Pipeline")
        print("=" * 60)
        print()
        print("Main Pipeline:")
        print("  1. Scraping - Collect news articles from websites")
        print("  2. Data Preparation - Complete processing pipeline")
        print("  3. Model Training - Train Bi-LSTM classification model")
        print("  4. Analysis - Analyze articles with trained model")
        print()
        print("Maintenance:")
        print("  9. Cleanup - Clean up done/ folders (except raw/done)")
        print()
        print("  0. Exit")
        print()
    
    def run(self):
        """Main pipeline runner"""
        print(f"🏠 Project Root: {self.project_root}")
        print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        while True:
            self.show_menu()
            
            try:
                choice = input("Select an option (0-4, 9): ").strip()
                
                if choice == '0':
                    print("👋 Goodbye!")
                    break
                elif choice == '1':
                    self.run_scraping()
                elif choice == '2':
                    self.prepare_data()
                elif choice == '3':
                    self.train_model()
                elif choice == '4':
                    self.analyze_articles()
                elif choice == '9':
                    self.cleanup_done_folders()
                else:
                    print("❌ Invalid option! Please select 0-4 or 9.")
                    continue
                
                input("\nPress Enter to continue...")
                
            except KeyboardInterrupt:
                print("\n👋 Pipeline interrupted by user. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                input("\nPress Enter to continue...")

def main():
    """Main entry point"""
    pipeline = BiLSTMPipeline()
    pipeline.run()

if __name__ == "__main__":
    main()