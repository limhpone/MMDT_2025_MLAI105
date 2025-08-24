#!/usr/bin/env python3
"""
Multi-Model Article Analyzer
Test multiple trained models on the same articles and compare performance
"""

import os
import pickle
import numpy as np
import pandas as pd
import json
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sys
import glob

# Add the myWord tokenizer path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '2_processor', 'tokenizer', 'myWord'))
from myword import MyWord

class MultiModelAnalyzer:
    def __init__(self, output_dir):
        """
        Initialize the multi-model analyzer
        
        Args:
            output_dir: Directory to save comparison outputs
        """
        self.output_dir = output_dir
        self.models = {}
        self.myword_tokenizer = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load all available models
        self._discover_models()
        
        # Initialize Myanmar tokenizer
        self._initialize_myword()
    
    def _discover_models(self):
        """Discover all trained models in the final_model directory"""
        print("Discovering available models...")
        
        # Get final model directory
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        from utils import get_data_directories
        dirs = get_data_directories()
        final_model_base_dir = dirs['final_model']
        
        # Find all model directories
        model_dirs = []
        if os.path.exists(final_model_base_dir):
            # Check for new structure (subdirectories)
            for item in os.listdir(final_model_base_dir):
                item_path = os.path.join(final_model_base_dir, item)
                if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, 'bilstm_model.h5')):
                    model_dirs.append(item_path)
            
            # Check for old structure (direct files)
            if os.path.exists(os.path.join(final_model_base_dir, 'bilstm_model.h5')):
                model_dirs.append(final_model_base_dir)
        
        if not model_dirs:
            print("❌ No trained models found in final_model directory")
            return
        
        # Load each model
        for model_dir in model_dirs:
            model_name = os.path.basename(model_dir)
            if model_name == os.path.basename(final_model_base_dir):  # Old structure
                model_name = "legacy_model"
            
            try:
                model_info = self._load_single_model(model_dir, model_name)
                self.models[model_name] = model_info
                print(f"✅ Loaded model: {model_name} (accuracy: {model_info.get('accuracy', 'unknown')})")
            except Exception as e:
                print(f"❌ Failed to load model {model_name}: {e}")
        
        print(f"📊 Discovered {len(self.models)} models for comparison")
    
    def _load_single_model(self, model_dir, model_name):
        """Load a single model and its artifacts"""
        # Load model
        model_path = os.path.join(model_dir, 'bilstm_model.h5')
        model = load_model(model_path)
        
        # Load tokenizer
        tokenizer_path = os.path.join(model_dir, 'tokenizer.pickle')
        with open(tokenizer_path, 'rb') as f:
            tokenizer_data = pickle.load(f)
        
        # Handle different tokenizer formats
        if hasattr(tokenizer_data, 'word_index'):
            # New SimpleTokenizer class format
            tokenizer = tokenizer_data
        elif isinstance(tokenizer_data, dict) and 'word_index' in tokenizer_data:
            # Dictionary format - create compatible wrapper
            class CompatibleTokenizer:
                def __init__(self, word_index):
                    self.word_index = word_index
                
                def texts_to_sequences(self, texts):
                    sequences = []
                    for text in texts:
                        if isinstance(text, str):
                            tokens = text.strip().split()
                        else:
                            tokens = text
                        sequence = [self.word_index.get(token, 1) for token in tokens]  # 1 is OOV
                        sequences.append(sequence)
                    return sequences
            
            tokenizer = CompatibleTokenizer(tokenizer_data['word_index'])
        else:
            # Old format
            tokenizer = tokenizer_data
        
        # Load model parameters
        params_path = os.path.join(model_dir, 'model_params.pickle')
        with open(params_path, 'rb') as f:
            model_params = pickle.load(f)
        
        # Load model info if available
        model_info_path = os.path.join(model_dir, 'model_info.json')
        model_info = {}
        if os.path.exists(model_info_path):
            with open(model_info_path, 'r') as f:
                model_info = json.load(f)
        
        return {
            'model': model,
            'tokenizer': tokenizer,
            'model_params': model_params,
            'model_info': model_info,
            'model_dir': model_dir,
            'accuracy': model_info.get('accuracy', 'unknown')
        }
    
    def _initialize_myword(self):
        """Initialize myWord tokenizer"""
        print("Initializing myWord tokenizer...")
        try:
            self.myword_tokenizer = MyWord()
            if self.myword_tokenizer.initialized:
                print("✅ myWord tokenizer initialized successfully")
            else:
                print("⚠️ Warning: myWord tokenizer failed to initialize")
        except Exception as e:
            print(f"❌ Error initializing myWord: {e}")
            self.myword_tokenizer = None
    
    def clean_text(self, text):
        """Basic text cleaning"""
        import re
        import unicodedata
        
        # Normalize Unicode
        text = unicodedata.normalize('NFC', text)
        
        # Remove unwanted characters while preserving Myanmar text
        pattern = r'[^\u1000-\u109F\u0020-\u007E\u00A0-\u00FF\uAA60-\uAA7F\uA9E0-\uA9FF\u2000-\u206F\u2070-\u209F\u20A0-\u20CF]'
        text = re.sub(pattern, ' ', text)
        
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def tokenize_text(self, text):
        """Tokenize text using myWord (similar to training pipeline)"""
        if not self.myword_tokenizer or not self.myword_tokenizer.initialized:
            # Fallback to simple whitespace tokenization
            print("Warning: Using fallback tokenization")
            return text.split()
        
        try:
            tokens = self.myword_tokenizer.segment(text)
            return tokens
        except Exception as e:
            print(f"Error in myWord tokenization: {e}")
            return text.split()
    
    def preprocess_article_for_model(self, text, model_name):
        """Preprocess article text for model input (matching original analyzer exactly)"""
        import re
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model_info = self.models[model_name]
        tokenizer = model_info['tokenizer']
        model_params = model_info['model_params']
        
        # Always apply myword tokenization to ensure consistency with training data
        # The training data uses myword syllable-level tokenization, so we must apply it here too
        cleaned_text = self.clean_text(text)
        tokens = self.tokenize_text(cleaned_text)
        tokens_text = ' '.join(tokens)
        
        # Debug: Print tokenization info
        print(f"Original text sample: {cleaned_text[:100]}...")
        print(f"Tokenized sample: {tokens_text[:100]}...")
        print(f"Token count: {len(tokens)}")
        
        # Convert to sequence using training tokenizer
        sequence = tokenizer.texts_to_sequences([tokens_text])
        
        # Pad sequence
        padded_sequence = pad_sequences(
            sequence, 
            maxlen=model_params['max_length'], 
            padding='post', 
            truncating='post'
        )
        
        return padded_sequence, tokens, tokens_text
    
    def predict_with_model(self, text, model_name):
        """Predict article category using a specific model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model_info = self.models[model_name]
        model = model_info['model']
        model_params = model_info['model_params']
        
        # Preprocess text (matching original analyzer exactly)
        padded_sequence, tokens, tokens_text = self.preprocess_article_for_model(text, model_name)
        
        # Predict
        prediction_probs = model.predict(padded_sequence, verbose=0)[0]
        predicted_class = np.argmax(prediction_probs)
        
        # Get class name
        label_mapping = model_params.get('label_mapping', {0: 'red', 1: 'neutral', 2: 'green'})
        predicted_label = label_mapping[predicted_class]
        
        return {
            'predicted_class': int(predicted_class),
            'predicted_label': predicted_label,
            'probabilities': {
                'red': float(prediction_probs[0]),
                'neutral': float(prediction_probs[1]),
                'green': float(prediction_probs[2])
            },
            'confidence': float(np.max(prediction_probs)),
            'token_count': len(tokens)
        }
    
    def analyze_article_with_all_models(self, article_path):
        """Analyze a single article with all available models"""
        print(f"\n📄 Analyzing: {os.path.basename(article_path)}")
        
        # Read article
        with open(article_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        results = {
            'article_path': article_path,
            'article_name': os.path.basename(article_path),
            'content_length': len(content),
            'analysis_time': datetime.now().isoformat(),
            'model_predictions': {}
        }
        
        # Test with each model
        for model_name, model_info in self.models.items():
            try:
                prediction = self.predict_with_model(content, model_name)
                prediction['model_accuracy'] = model_info.get('accuracy', 'unknown')
                prediction['model_info'] = model_info.get('model_info', {})
                results['model_predictions'][model_name] = prediction
                
                print(f"  {model_name}: {prediction['predicted_label'].upper()} (conf: {prediction['confidence']:.3f})")
                
            except Exception as e:
                print(f"  ❌ {model_name}: Error - {e}")
                results['model_predictions'][model_name] = {'error': str(e)}
        
        return results
    
    def compare_models_on_test_data(self):
        """Compare all models on test articles following raw→processed→done workflow or using processed files if called from pipeline"""
        print("\n" + "="*60)
        print("MULTI-MODEL ARTICLE CLASSIFICATION COMPARISON")
        print("="*60)
        
        if not self.models:
            print("❌ No models available for comparison")
            return
        
        # Get directories
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        from utils import get_data_directories
        dirs = get_data_directories()
        
        # Check if we're being called from pipeline with processed files
        processed_dir_override = os.environ.get('PROCESSED_DIR')
        if processed_dir_override:
            print(f"📁 Using processed directory from pipeline: {processed_dir_override}")
            test_dir = processed_dir_override
            workflow_mode = "pipeline"
        else:
            print(f"📁 Using standalone mode: raw→processed→done workflow")
            test_dir = dirs['model_tester_raw']
            workflow_mode = "standalone"
        
        # Step 1: Check for files in test directory
        test_files = []
        for ext in ['*.txt', '*.md']:
            test_files.extend(glob.glob(os.path.join(test_dir, ext)))
        
        if not test_files:
            print(f"❌ No test articles found in {test_dir}")
            print(f"📝 Please add .txt or .md files to: {test_dir}")
            return
        
        print(f"📂 Found {len(test_files)} articles in test directory")
        print(f"🤖 Testing with {len(self.models)} models")
        
        if workflow_mode == "pipeline":
            print("📝 Note: Using pre-tokenized files from pipeline")
        else:
            print("📝 Note: Following complete raw→processed→done workflow")
        
        # Step 2: Handle workflow based on mode
        if workflow_mode == "standalone":
            all_results = self._process_standalone_workflow(test_files, dirs)
        else:
            all_results = self._process_pipeline_workflow(test_files)
        
        if not all_results:
            print("❌ No results to process")
            return
        
        # Step 3: Generate comparison report
        self.generate_comparison_report(all_results)
    
    def _process_standalone_workflow(self, raw_files, dirs):
        """Process files using standalone workflow: raw→processed→done"""
        import shutil
        
        raw_dir = dirs['model_tester_raw'] 
        processed_dir = dirs['model_tester_processed']
        done_dir = dirs['model_tester_done']
        
        print(f"📄 Following workflow: raw/ → processed/ → done/")
        
        # Create directories
        os.makedirs(processed_dir, exist_ok=True)
        os.makedirs(done_dir, exist_ok=True)
        
        all_results = []
        processed_files = []
        
        try:
            # Step 1: Tokenize raw files and save to processed
            print(f"🔤 Step 1: Tokenizing files from raw/ to processed/")
            for file_path in raw_files:
                filename = os.path.basename(file_path)
                print(f"   📋 Tokenizing: {filename}")
                
                # Read original file
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                
                # Tokenize content using myWord
                cleaned_text = self.clean_text(content)
                tokens = self.tokenize_text(cleaned_text)
                tokens_text = ' '.join(tokens)
                
                # Save tokenized version to processed directory
                processed_path = os.path.join(processed_dir, filename)
                with open(processed_path, 'w', encoding='utf-8') as f:
                    f.write(tokens_text)
                
                processed_files.append(processed_path)
                print(f"      ✅ Tokenized (Token count: {len(tokens)})")
            
            # Step 2: Analyze processed files
            print(f"🤖 Step 2: Analyzing processed files")
            for processed_path in processed_files:
                filename = os.path.basename(processed_path)
                print(f"   🔍 Analyzing: {filename}")
                
                try:
                    result = self.analyze_article_with_all_models(processed_path)
                    all_results.append(result)
                except Exception as e:
                    print(f"      ❌ Error processing {filename}: {e}")
            
            # Step 3: Move processed files to done
            print(f"📦 Step 3: Moving processed files to done/")
            for processed_path in processed_files:
                filename = os.path.basename(processed_path)
                done_path = os.path.join(done_dir, filename)
                if os.path.exists(processed_path):
                    shutil.move(processed_path, done_path)
                    print(f"      ✅ Moved: {filename}")
                    
        except Exception as e:
            print(f"❌ Error in standalone workflow: {e}")
            return []
        
        return all_results
    
    def _process_pipeline_workflow(self, processed_files):
        """Process pre-tokenized files from pipeline"""
        all_results = []
        
        # Analyze processed files directly (already tokenized by pipeline)
        for file_path in processed_files:
            filename = os.path.basename(file_path)
            print(f"🔍 Analyzing: {filename}")
            
            try:
                result = self.analyze_article_with_all_models(file_path)
                all_results.append(result)
            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")
        
        return all_results
    
    def generate_comparison_report(self, results):
        """Generate a comprehensive comparison report"""
        print("\n" + "="*60)
        print("MODEL COMPARISON SUMMARY")
        print("="*60)
        
        if not results:
            return
        
        # Initialize comparison data
        model_names = list(self.models.keys())
        comparison_data = {
            'total_articles': len(results),
            'models_compared': len(model_names),
            'model_accuracies': {name: self.models[name].get('accuracy', 'unknown') for name in model_names},
            'prediction_summary': {},
            'agreement_analysis': {},
            'confidence_analysis': {}
        }
        
        # Analyze predictions
        for model_name in model_names:
            predictions = []
            confidences = []
            
            for result in results:
                if model_name in result['model_predictions']:
                    pred = result['model_predictions'][model_name]
                    if 'predicted_label' in pred:
                        predictions.append(pred['predicted_label'])
                        confidences.append(pred['confidence'])
            
            comparison_data['prediction_summary'][model_name] = {
                'total_predictions': len(predictions),
                'red_predictions': predictions.count('red'),
                'neutral_predictions': predictions.count('neutral'),
                'green_predictions': predictions.count('green'),
                'avg_confidence': np.mean(confidences) if confidences else 0
            }
        
        # Print summary
        print(f"\n📊 Model Performance Summary:")
        print(f"{'Model Name':<25} {'Accuracy':<12} {'Avg Confidence':<15} {'Red':<5} {'Neutral':<8} {'Green':<6}")
        print("-" * 80)
        
        for model_name in model_names:
            summary = comparison_data['prediction_summary'][model_name]
            accuracy = comparison_data['model_accuracies'][model_name]
            acc_str = f"{accuracy:.4f}" if isinstance(accuracy, (int, float)) else str(accuracy)
            
            print(f"{model_name:<25} {acc_str:<12} {summary['avg_confidence']:<15.3f} "
                  f"{summary['red_predictions']:<5} {summary['neutral_predictions']:<8} "
                  f"{summary['green_predictions']:<6}")
        
        # Create session-based output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        session_dir = os.path.join(self.output_dir, f'session_{timestamp}')
        os.makedirs(session_dir, exist_ok=True)
        
        # Save detailed report in session folder
        report_path = os.path.join(session_dir, 'multi_model_comparison.json')
        
        with open(report_path, 'w') as f:
            json.dump({
                'session_timestamp': timestamp,
                'comparison_data': comparison_data,
                'detailed_results': results
            }, f, indent=2)
        
        print(f"\n📋 Detailed comparison report saved: {report_path}")
        
        # Generate markdown report in session folder
        md_report_path = os.path.join(session_dir, 'multi_model_comparison.md')
        self._generate_markdown_report(comparison_data, results, md_report_path)
        print(f"📄 Markdown report saved: {md_report_path}")
        print(f"📁 Session directory: {session_dir}")
    
    def _generate_markdown_report(self, comparison_data, results, report_path):
        """Generate markdown comparison report"""
        md_content = f"""# Multi-Model Performance Comparison

Generated: {datetime.now().isoformat()}

## Summary

- **Total Articles Tested**: {comparison_data['total_articles']}
- **Models Compared**: {comparison_data['models_compared']}

## Model Performance Overview

| Model Name | Training Accuracy | Avg Confidence | Red Predictions | Neutral Predictions | Green Predictions |
|------------|------------------|-----------------|-----------------|-------------------|------------------|
"""
        
        for model_name in comparison_data['prediction_summary']:
            summary = comparison_data['prediction_summary'][model_name]
            accuracy = comparison_data['model_accuracies'][model_name]
            acc_str = f"{accuracy:.4f}" if isinstance(accuracy, (int, float)) else str(accuracy)
            
            md_content += f"| {model_name} | {acc_str} | {summary['avg_confidence']:.3f} | "
            md_content += f"{summary['red_predictions']} | {summary['neutral_predictions']} | "
            md_content += f"{summary['green_predictions']} |\n"
        
        md_content += "\n## Individual Article Results\n\n"
        
        for result in results:
            md_content += f"### {result['article_name']}\n\n"
            md_content += "| Model | Prediction | Confidence | Red Prob | Neutral Prob | Green Prob |\n"
            md_content += "|-------|------------|------------|----------|--------------|------------|\n"
            
            for model_name, pred in result['model_predictions'].items():
                if 'predicted_label' in pred:
                    md_content += f"| {model_name} | {pred['predicted_label'].upper()} | "
                    md_content += f"{pred['confidence']:.3f} | {pred['probabilities']['red']:.3f} | "
                    md_content += f"{pred['probabilities']['neutral']:.3f} | "
                    md_content += f"{pred['probabilities']['green']:.3f} |\n"
                else:
                    md_content += f"| {model_name} | ERROR | - | - | - | - |\n"
            
            md_content += "\n"
        
        with open(report_path, 'w') as f:
            f.write(md_content)

def main():
    """Main function"""
    # Setup directories
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from utils import get_data_directories
    
    dirs = get_data_directories()
    # Use a simple output directory in 4_analyzer
    output_dir = os.path.join(os.path.dirname(__file__), 'multi_model_comparisons')
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize analyzer
    analyzer = MultiModelAnalyzer(output_dir)
    
    if not analyzer.models:
        print("❌ No models available for testing. Please train some models first.")
        return
    
    # Run comparison (no test_dir parameter - uses raw→processed→done workflow)
    results = analyzer.compare_models_on_test_data()
    
    print("\n🎉 Multi-model comparison completed!")

if __name__ == "__main__":
    main()