import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sys
import shutil

# Add the myWord tokenizer path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '2_processor', 'tokenizer', 'myWord'))
from myword import MyWord

class MyanmarArticleAnalyzer:
    def __init__(self, model_dir, output_dir):
        """
        Initialize the article analyzer
        
        Args:
            model_dir: Directory containing trained model and artifacts
            output_dir: Directory to save analysis outputs
        """
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.model = None
        self.tokenizer = None
        self.myword_tokenizer = None
        self.model_params = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load model and artifacts
        self._load_model_artifacts()
        
        # Initialize Myanmar tokenizer
        self._initialize_myword()
    
    def _load_model_artifacts(self):
        """Load trained model and preprocessing artifacts"""
        print("Loading model and artifacts...")
        
        # First, check if there's a final model directory
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        from utils import get_data_directories
        dirs = get_data_directories()
        final_model_dir = dirs['final_model']
        
        # Check if final model exists
        final_model_path = os.path.join(final_model_dir, 'bilstm_model.h5')
        if os.path.exists(final_model_path):
            print(f"🤖 Using final production model from: {final_model_dir}")
            self.model_dir = final_model_dir
            model_path = final_model_path
            
            # Load tokenizer and parameters from final model directory
            tokenizer_path = os.path.join(final_model_dir, 'tokenizer.pickle')
            params_path = os.path.join(final_model_dir, 'model_params.pickle')
            
            with open(tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
            print(f"Tokenizer loaded from: {tokenizer_path}")
            
            with open(params_path, 'rb') as f:
                self.model_params = pickle.load(f)
            print(f"Model parameters loaded from: {params_path}")
            
            self.model = load_model(model_path)
            print(f"Model loaded from: {model_path}")
            
            print(f"Vocabulary size: {self.model_params['vocab_size']}")
            print(f"Max sequence length: {self.model_params['max_length']}")
            print(f"Label mapping: {self.model_params['label_mapping']}")
            return
        
        # If no final model, fall back to training models
        print(f"🤖 No final model found, checking training models...")
        
        # Find the latest model
        model_files = [f for f in os.listdir(self.model_dir) if f.startswith('bilstm_model_') and f.endswith('.h5')]
        if model_files:
            # Sort by timestamp and get the latest
            model_files.sort(reverse=True)
            latest_model = model_files[0]
            model_path = os.path.join(self.model_dir, latest_model)
            print(f"🤖 Using latest training model: {latest_model}")
            
            # Extract timestamp from model filename
            timestamp = latest_model.replace('bilstm_model_', '').replace('.h5', '')
            
            # Look for corresponding session reports directory
            training_reports_dir = os.path.join(self.model_dir, 'training_reports')
            if os.path.exists(training_reports_dir):
                session_dirs = [d for d in os.listdir(training_reports_dir) if d.startswith('training_report_')]
                if session_dirs:
                    # Find the session directory that matches the model timestamp
                    matching_session = None
                    for session_dir in session_dirs:
                        session_timestamp = session_dir.replace('training_report_', '')
                        if session_timestamp == timestamp:
                            matching_session = session_dir
                            break
                    
                    if matching_session:
                        session_path = os.path.join(training_reports_dir, matching_session)
                        print(f"📁 Found matching session: {matching_session}")
                        
                        # Load tokenizer from session directory
                        tokenizer_path = os.path.join(session_path, 'tokenizer.pickle')
                        if os.path.exists(tokenizer_path):
                            with open(tokenizer_path, 'rb') as f:
                                self.tokenizer = pickle.load(f)
                            print(f"Tokenizer loaded from: {tokenizer_path}")
                        else:
                            # Fallback to main directory
                            tokenizer_path = os.path.join(self.model_dir, 'tokenizer.pickle')
                            with open(tokenizer_path, 'rb') as f:
                                self.tokenizer = pickle.load(f)
                            print(f"Tokenizer loaded from: {tokenizer_path}")
                        
                        # Load model parameters from session directory
                        params_path = os.path.join(session_path, 'model_params.pickle')
                        if os.path.exists(params_path):
                            with open(params_path, 'rb') as f:
                                self.model_params = pickle.load(f)
                            print(f"Model parameters loaded from: {params_path}")
                        else:
                            # Fallback to main directory
                            params_path = os.path.join(self.model_dir, 'model_params.pickle')
                            with open(params_path, 'rb') as f:
                                self.model_params = pickle.load(f)
                            print(f"Model parameters loaded from: {params_path}")
                    else:
                        # Fallback to main directory
                        tokenizer_path = os.path.join(self.model_dir, 'tokenizer.pickle')
                        with open(tokenizer_path, 'rb') as f:
                            self.tokenizer = pickle.load(f)
                        print(f"Tokenizer loaded from: {tokenizer_path}")
                        
                        params_path = os.path.join(self.model_dir, 'model_params.pickle')
                        with open(params_path, 'rb') as f:
                            self.model_params = pickle.load(f)
                        print(f"Model parameters loaded from: {params_path}")
                else:
                    # Fallback to main directory
                    tokenizer_path = os.path.join(self.model_dir, 'tokenizer.pickle')
                    with open(tokenizer_path, 'rb') as f:
                        self.tokenizer = pickle.load(f)
                    print(f"Tokenizer loaded from: {tokenizer_path}")
                    
                    params_path = os.path.join(self.model_dir, 'model_params.pickle')
                    with open(params_path, 'rb') as f:
                        self.model_params = pickle.load(f)
                    print(f"Model parameters loaded from: {params_path}")
            else:
                # Fallback to main directory
                tokenizer_path = os.path.join(self.model_dir, 'tokenizer.pickle')
                with open(tokenizer_path, 'rb') as f:
                    self.tokenizer = pickle.load(f)
                print(f"Tokenizer loaded from: {tokenizer_path}")
                
                params_path = os.path.join(self.model_dir, 'model_params.pickle')
                with open(params_path, 'rb') as f:
                    self.model_params = pickle.load(f)
                print(f"Model parameters loaded from: {params_path}")
        else:
            # Fallback to legacy model name
            model_path = os.path.join(self.model_dir, 'bilstm_model.h5')
            print(f"🤖 Using legacy model: bilstm_model.h5")
            
            # Load from main directory for legacy model
            tokenizer_path = os.path.join(self.model_dir, 'tokenizer.pickle')
            with open(tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
            print(f"Tokenizer loaded from: {tokenizer_path}")
            
            params_path = os.path.join(self.model_dir, 'model_params.pickle')
            with open(params_path, 'rb') as f:
                self.model_params = pickle.load(f)
            print(f"Model parameters loaded from: {params_path}")
        
        self.model = load_model(model_path)
        print(f"Model loaded from: {model_path}")
        
        print(f"Vocabulary size: {self.model_params['vocab_size']}")
        print(f"Max sequence length: {self.model_params['max_length']}")
        print(f"Label mapping: {self.model_params['label_mapping']}")
    
    def _initialize_myword(self):
        """Initialize myWord tokenizer"""
        print("Initializing myWord tokenizer...")
        try:
            self.myword_tokenizer = MyWord()
            if self.myword_tokenizer.initialized:
                print("myWord tokenizer initialized successfully")
            else:
                print("Warning: myWord tokenizer failed to initialize")
        except Exception as e:
            print(f"Error initializing myWord: {e}")
            self.myword_tokenizer = None
    
    def clean_text(self, text):
        """Basic text cleaning similar to training pipeline"""
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
        
        # Remove excessive punctuation
        text = re.sub(r'[,.!?;:]{2,}', '.', text)
        text = re.sub(r'\.{3,}', '...', text)
        
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
    
    def preprocess_article(self, text):
        """Preprocess article text for model input"""
        import re
        
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
        sequence = self.tokenizer.texts_to_sequences([tokens_text])
        
        # Pad sequence
        padded_sequence = pad_sequences(
            sequence, 
            maxlen=self.model_params['max_length'], 
            padding='post', 
            truncating='post'
        )
        
        return padded_sequence, tokens, tokens_text
    
    def predict_article(self, text):
        """Predict article category"""
        # Preprocess
        padded_sequence, tokens, tokens_text = self.preprocess_article(text)
        
        # Predict
        prediction_probs = self.model.predict(padded_sequence, verbose=0)[0]
        predicted_class = np.argmax(prediction_probs)
        predicted_label = self.model_params['label_mapping'][predicted_class]
        confidence = float(prediction_probs[predicted_class])
        
        return {
            'predicted_class': int(predicted_class),
            'predicted_label': predicted_label,
            'confidence': confidence,
            'probabilities': {
                self.model_params['label_mapping'][i]: float(prob) 
                for i, prob in enumerate(prediction_probs)
            },
            'tokens': tokens,
            'tokens_text': tokens_text,
            'token_count': len(tokens)
        }
    
    def analyze_single_article(self, file_path):
        """Analyze a single article file"""
        print(f"Analyzing: {os.path.basename(file_path)}")
        
        # Read article
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # Get expected label from filename
        filename = os.path.basename(file_path)
        if filename.startswith('red_'):
            expected_label = 'red'
        elif filename.startswith('green_'):
            expected_label = 'green'
        elif filename.startswith('neutral_'):
            expected_label = 'neutral'
        else:
            expected_label = 'unknown'
        
        # Predict
        prediction_result = self.predict_article(content)
        
        # Calculate accuracy
        is_correct = prediction_result['predicted_label'] == expected_label
        
        return {
            'filename': filename,
            'expected_label': expected_label,
            'content': content,
            'content_length': len(content),
            'is_correct': is_correct,
            **prediction_result
        }
    
    def generate_text_report(self, results):
        """Generate comprehensive application-level analysis report"""
        report_path = os.path.join(self.output_dir, 'analysis_report.md')
        
        # Calculate detailed statistics
        total_articles = len(results)
        correct_predictions = sum(1 for r in results if r['is_correct'])
        accuracy = correct_predictions / total_articles if total_articles > 0 else 0
        
        # Category-wise analysis
        categories = ['red', 'neutral', 'green']
        category_stats = {cat: {'expected': 0, 'predicted': 0, 'correct': 0, 'confidences': []} 
                         for cat in categories}
        
        for result in results:
            exp_label = result['expected_label']
            pred_label = result['predicted_label']
            
            if exp_label in category_stats:
                category_stats[exp_label]['expected'] += 1
                category_stats[exp_label]['confidences'].append(result['confidence'])
                if result['is_correct']:
                    category_stats[exp_label]['correct'] += 1
            
            if pred_label in category_stats:
                category_stats[pred_label]['predicted'] += 1
        
        # Risk analysis
        high_risk_articles = [r for r in results if r['confidence'] < 0.7]
        misclassified_articles = [r for r in results if not r['is_correct']]
        
        with open(report_path, 'w', encoding='utf-8') as f:
            # Header
            f.write("# Myanmar Article Classification - Comprehensive Analysis Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Model:** {self.model_dir}\n")
            f.write(f"**Analyzed Articles:** {total_articles}\n\n")
            f.write("---\n\n")
            
            # EXECUTIVE SUMMARY
            f.write("## 🔴 Executive Summary\n\n")
            f.write(f"- **Overall Model Accuracy:** {accuracy:.1%}\n")
            f.write(f"- **Articles Correctly Classified:** {correct_predictions}/{total_articles}\n")
            f.write(f"- **High-Risk Classifications** (confidence < 70%): {len(high_risk_articles)}\n")
            f.write(f"- **Misclassified Articles:** {len(misclassified_articles)}\n\n")
            
            if accuracy >= 0.9:
                f.write("**✅ STATUS: EXCELLENT** - Model performing very well\n\n")
            elif accuracy >= 0.8:
                f.write("**⚠️ STATUS: GOOD** - Model performing adequately\n\n")
            elif accuracy >= 0.7:
                f.write("**🔶 STATUS: MODERATE** - Model needs attention\n\n")
            else:
                f.write("**🔴 STATUS: CRITICAL** - Model requires immediate review\n\n")
            
            # KEY FINDINGS
            f.write("## 🎯 Key Findings\n\n")
            avg_confidence = np.mean([r['confidence'] for r in results])
            f.write(f"- **Average Prediction Confidence:** {avg_confidence:.1%}\n")
            
            best_category = max(category_stats.items(), 
                              key=lambda x: x[1]['correct']/max(x[1]['expected'], 1))[0]
            f.write(f"- **Best Performing Category:** {best_category.upper()}\n")
            
            if high_risk_articles:
                f.write(f"- **{len(high_risk_articles)} articles** flagged for manual review\n")
            
            f.write("\n")
            
            # CATEGORY PERFORMANCE
            f.write("## 📊 Category Performance Analysis\n\n")
            f.write("| Category | Expected | Predicted | Correct | Accuracy | Avg Confidence |\n")
            f.write("|----------|----------|-----------|---------|----------|----------------|\n")
            
            for category in categories:
                stats = category_stats[category]
                cat_accuracy = stats['correct'] / max(stats['expected'], 1)
                avg_conf = np.mean(stats['confidences']) if stats['confidences'] else 0
                
                f.write(f"| {category.upper()} | {stats['expected']} | {stats['predicted']} | {stats['correct']} | {cat_accuracy:.1%} | {avg_conf:.1%} |\n")
            f.write("\n")
            
            # CONFUSION MATRIX
            f.write("## 🔢 Confusion Matrix\n\n")
            f.write("| Actual \\ Predicted | Red | Neutral | Green |\n")
            f.write("|-------------------|-----|---------|-------|\n")
            
            for exp_cat in categories:
                counts = {'red': 0, 'neutral': 0, 'green': 0}
                for result in results:
                    if result['expected_label'] == exp_cat:
                        counts[result['predicted_label']] += 1
                
                f.write(f"| {exp_cat.capitalize()} | {counts['red']} | {counts['neutral']} | {counts['green']} |\n")
            f.write("\n")
            
            # RISK ASSESSMENT
            if high_risk_articles or misclassified_articles:
                f.write("## ⚠️ Risk Assessment & Flagged Articles\n\n")
                
                if high_risk_articles:
                    f.write(f"### 🔶 Low Confidence Predictions ({len(high_risk_articles)} articles)\n\n")
                    for article in high_risk_articles:
                        f.write(f"- **{article['filename']}**: {article['predicted_label']} "
                               f"({article['confidence']:.1%} confidence)\n")
                    f.write("\n")
                
                if misclassified_articles:
                    f.write(f"### ❌ Misclassified Articles ({len(misclassified_articles)} articles)\n\n")
                    for article in misclassified_articles:
                        f.write(f"- **{article['filename']}**: Expected {article['expected_label']}, "
                               f"Got {article['predicted_label']} ({article['confidence']:.1%})\n")
                    f.write("\n")
            
            # CONTENT ANALYSIS
            f.write("## 📝 Content Analysis Insights\n\n")
            
            # Token count analysis
            token_counts = [r['token_count'] for r in results]
            avg_tokens = np.mean(token_counts)
            f.write(f"- **Average Article Length:** {avg_tokens:.0f} tokens\n")
            f.write(f"- **Shortest Article:** {min(token_counts)} tokens ({min(results, key=lambda x: x['token_count'])['filename']})\n")
            f.write(f"- **Longest Article:** {max(token_counts)} tokens ({max(results, key=lambda x: x['token_count'])['filename']})\n\n")
            
            # Confidence distribution
            confidences = [r['confidence'] for r in results]
            f.write("### Confidence Score Distribution\n\n")
            f.write(f"- **Very High (90%+):** {sum(1 for c in confidences if c >= 0.9)} articles\n")
            f.write(f"- **High (80-89%):** {sum(1 for c in confidences if 0.8 <= c < 0.9)} articles\n")
            f.write(f"- **Medium (70-79%):** {sum(1 for c in confidences if 0.7 <= c < 0.8)} articles\n")
            f.write(f"- **Low (<70%):** {sum(1 for c in confidences if c < 0.7)} articles\n\n")
            
            # RECOMMENDATIONS
            f.write("## 💡 Recommendations & Action Items\n\n")
            
            if accuracy >= 0.95:
                f.write("✅ **Model performance is excellent.** Continue monitoring.\n")
            elif accuracy >= 0.85:
                f.write("⚡ **Consider fine-tuning** with additional data for edge cases.\n")
            else:
                f.write("🔴 **URGENT:** Model requires retraining or additional data.\n")
            
            if high_risk_articles:
                f.write(f"🔍 **Manual review recommended** for {len(high_risk_articles)} low-confidence predictions.\n")
            
            if len(set(r['expected_label'] for r in results)) < 3:
                f.write("📊 **Test with more diverse categories** to ensure robust evaluation.\n")
            
            f.write("📈 **Regular monitoring recommended** with production data.\n\n")
            
            # TECHNICAL DETAILS
            f.write("## 🔧 Technical Details\n\n")
            f.write(f"- **Model Architecture:** Bidirectional LSTM\n")
            f.write(f"- **Vocabulary Size:** {self.model_params['vocab_size']:,}\n")
            f.write(f"- **Max Sequence Length:** {self.model_params['max_length']}\n")
            f.write(f"- **Categories:** {', '.join(categories)}\n")
            f.write(f"- **Tokenization:** Myanmar myWord segmentation\n\n")
            
            # DETAILED ARTICLE ANALYSIS
            f.write("## 📋 Detailed Article Analysis\n\n")
            
            for i, result in enumerate(results, 1):
                status = "✅ CORRECT" if result['is_correct'] else "❌ INCORRECT"
                risk_level = "🔴 HIGH RISK" if result['confidence'] < 0.7 else "✅ LOW RISK"
                
                f.write(f"### Article {i}: {result['filename']}\n\n")
                f.write(f"**Classification:** {result['expected_label'].upper()} → {result['predicted_label'].upper()} | {status}\n")
                f.write(f"**Confidence:** {result['confidence']:.1%} | {risk_level}\n")
                f.write(f"**Content:** {result['content_length']} chars, {result['token_count']} tokens\n\n")
                
                f.write("**Prediction Probabilities:**\n\n")
                sorted_probs = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)
                f.write("| Label | Probability | Visualization |\n")
                f.write("|-------|-------------|---------------|\n")
                for label, prob in sorted_probs:
                    bar_length = int(prob * 10)  # Shorter bars for markdown table
                    bar = "█" * bar_length + "░" * (10 - bar_length)
                    f.write(f"| {label.upper()} | {prob:.1%} | `{bar}` |\n")
                f.write("\n")
                
                f.write("**Content Preview:**\n\n")
                f.write("```\n")
                f.write(f"{result['content'][:400]}{'...' if len(result['content']) > 400 else ''}\n")
                f.write("```\n\n")
                f.write("---\n\n")
        
        print(f"Comprehensive analysis report saved to: {report_path}")
        return report_path
    
    def generate_visual_report(self, results):
        """Generate visual analysis report"""
        print("Generating visual analysis report...")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Confidence scores comparison
        ax1 = plt.subplot(2, 3, 1)
        filenames = [r['filename'].replace('_article.txt', '') for r in results]
        confidences = [r['confidence'] for r in results]
        colors = ['green' if r['is_correct'] else 'red' for r in results]
        
        bars = ax1.bar(filenames, confidences, color=colors, alpha=0.7)
        ax1.set_title('Prediction Confidence by Article')
        ax1.set_ylabel('Confidence Score')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, conf in zip(bars, confidences):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{conf:.3f}', ha='center', va='bottom')
        
        # 2. Probability distribution heatmap
        ax2 = plt.subplot(2, 3, 2)
        prob_matrix = []
        for result in results:
            probs = [result['probabilities']['red'], 
                    result['probabilities']['neutral'], 
                    result['probabilities']['green']]
            prob_matrix.append(probs)
        
        prob_df = pd.DataFrame(prob_matrix, 
                              columns=['Red', 'Neutral', 'Green'],
                              index=filenames)
        
        sns.heatmap(prob_df, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                   ax=ax2, cbar_kws={'label': 'Probability'})
        ax2.set_title('Prediction Probabilities Heatmap')
        ax2.set_ylabel('Articles')
        
        # 3. Expected vs Predicted
        ax3 = plt.subplot(2, 3, 3)
        expected_labels = [r['expected_label'] for r in results]
        predicted_labels = [r['predicted_label'] for r in results]
        
        # Create confusion-like visualization
        label_names = ['red', 'neutral', 'green']
        comparison_data = []
        for exp, pred in zip(expected_labels, predicted_labels):
            comparison_data.append([exp, pred])
        
        comparison_df = pd.DataFrame(comparison_data, columns=['Expected', 'Predicted'])
        comparison_counts = comparison_df.groupby(['Expected', 'Predicted']).size().unstack(fill_value=0)
        
        if not comparison_counts.empty:
            sns.heatmap(comparison_counts, annot=True, fmt='d', cmap='Blues', ax=ax3)
        ax3.set_title('Expected vs Predicted Labels')
        
        # 4. Token count analysis
        ax4 = plt.subplot(2, 3, 4)
        token_counts = [r['token_count'] for r in results]
        ax4.bar(filenames, token_counts, color='skyblue', alpha=0.7)
        ax4.set_title('Token Count by Article')
        ax4.set_ylabel('Number of Tokens')
        ax4.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for i, count in enumerate(token_counts):
            ax4.text(i, count + max(token_counts) * 0.02, str(count), 
                    ha='center', va='bottom')
        
        # 5. Content length analysis
        ax5 = plt.subplot(2, 3, 5)
        content_lengths = [r['content_length'] for r in results]
        ax5.bar(filenames, content_lengths, color='lightcoral', alpha=0.7)
        ax5.set_title('Content Length by Article')
        ax5.set_ylabel('Characters')
        ax5.tick_params(axis='x', rotation=45)
        
        # 6. Accuracy summary
        ax6 = plt.subplot(2, 3, 6)
        correct_count = sum(1 for r in results if r['is_correct'])
        incorrect_count = len(results) - correct_count
        
        labels = ['Correct', 'Incorrect']
        sizes = [correct_count, incorrect_count]
        colors = ['lightgreen', 'lightcoral']
        
        ax6.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax6.set_title('Overall Accuracy')
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(self.output_dir, 'visual_analysis_report.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Visual report saved to: {plot_path}")
        plt.show()
        
        return plot_path
    
    def analyze_all_articles(self, test_dir):
        """Analyze all articles in the test directory"""
        print("=" * 60)
        print("MYANMAR ARTICLE CLASSIFICATION ANALYSIS")
        print("=" * 60)
        
        # Find all article files
        article_files = []
        for file in os.listdir(test_dir):
            if file.endswith('.txt'):
                article_files.append(os.path.join(test_dir, file))
        
        if not article_files:
            print(f"No article files found in {test_dir}")
            return
        
        article_files.sort()
        print(f"Found {len(article_files)} articles to analyze\\n")
        
        # Analyze each article
        results = []
        for file_path in article_files:
            result = self.analyze_single_article(file_path)
            results.append(result)
        
        # Generate reports
        print("\\nGenerating analysis reports...")
        text_report_path = self.generate_text_report(results)
        visual_report_path = self.generate_visual_report(results)
        
        # Print summary
        print("\\n" + "=" * 60)
        print("ANALYSIS SUMMARY")
        print("=" * 60)
        
        total_articles = len(results)
        correct_predictions = sum(1 for r in results if r['is_correct'])
        accuracy = correct_predictions / total_articles if total_articles > 0 else 0
        
        print(f"Total articles: {total_articles}")
        print(f"Correct predictions: {correct_predictions}")
        print(f"Overall accuracy: {accuracy:.2%}")
        print(f"\\nReports saved to: {self.output_dir}")
        print(f"- Text report: {os.path.basename(text_report_path)}")
        print(f"- Visual report: {os.path.basename(visual_report_path)}")
        
        return results

def main():
    """Main analysis function"""
    # Import project utilities
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from utils import get_data_directories
    
    # Get clean directory paths
    dirs = get_data_directories()
    model_dir = dirs['model_output']
    
    # Check if we should use a specific processed directory (from pipeline)
    test_dir = os.environ.get('PROCESSED_DIR')
    if not test_dir:
        test_dir = dirs['model_tester_done']  # Use done folder for testing
    
    output_dir = dirs['analysis_output']
    
    # Create raw directory if it doesn't exist
    raw_dir = dirs['model_tester_raw']
    os.makedirs(raw_dir, exist_ok=True)
    print(f"📁 Raw directory created/verified: {raw_dir}")
    
    # Find the latest model
    model_files = [f for f in os.listdir(model_dir) if f.startswith('bilstm_model_') and f.endswith('.h5')]
    if model_files:
        # Sort by timestamp and get the latest
        model_files.sort(reverse=True)
        latest_model = model_files[0]
        model_path = os.path.join(model_dir, latest_model)
        print(f"🤖 Using latest model: {latest_model}")
    else:
        # Fallback to legacy model name
        model_path = os.path.join(model_dir, 'bilstm_model.h5')
        if not os.path.exists(model_path):
            print(f"Error: No trained model found in {model_dir}")
            print("Please train the model first using the trainer.")
            return
        print(f"🤖 Using legacy model: bilstm_model.h5")
    
    # Check if test articles exist
    if not os.path.exists(test_dir):
        print(f"Error: Test directory not found: {test_dir}")
        return
    
    # Create session-based output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    session_output_dir = os.path.join(output_dir, f"analysis_{timestamp}")
    os.makedirs(session_output_dir, exist_ok=True)
    
    # Initialize analyzer
    analyzer = MyanmarArticleAnalyzer(model_dir, session_output_dir)
    
    # Run analysis
    results = analyzer.analyze_all_articles(test_dir)
    
    print("\\nAnalysis complete!")

if __name__ == "__main__":
    main()