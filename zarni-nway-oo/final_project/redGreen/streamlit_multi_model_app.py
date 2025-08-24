#!/usr/bin/env python3
"""
Myanmar Article Classification - Multi-Model Streamlit Web App
Compare predictions from multiple trained models
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import unicodedata
from datetime import datetime
import json

# Add the myWord tokenizer path
sys.path.append(os.path.join(os.path.dirname(__file__), '2_processor', 'tokenizer', 'myWord'))

# Page configuration
st.set_page_config(
    page_title="Myanmar Multi-Model Article Classifier",
    page_icon="🇲🇲",
    layout="wide",
    initial_sidebar_state="expanded"
)

class MultiModelClassifier:
    def __init__(self):
        """Initialize the multi-model classifier"""
        self.models = {}
        self.myword_tokenizer = None
        
        # Load all available models
        self._discover_and_load_models()
        
        # Initialize Myanmar tokenizer
        self._initialize_myword()
    
    def _discover_and_load_models(self):
        """Discover and load all trained models"""
        st.write("🔍 Discovering available models...")
        
        # Get final model directory
        sys.path.append('.')
        from utils import get_data_directories
        dirs = get_data_directories()
        final_model_base_dir = dirs['final_model']
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
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
            st.error("❌ No trained models found in final_model directory")
            return
        
        # Load each model
        for i, model_dir in enumerate(model_dirs):
            model_name = os.path.basename(model_dir)
            if model_name == os.path.basename(final_model_base_dir):  # Old structure
                model_name = "legacy_model"
            
            try:
                status_text.text(f"Loading {model_name}...")
                model_info = self._load_single_model(model_dir, model_name)
                self.models[model_name] = model_info
                
                progress_bar.progress((i + 1) / len(model_dirs))
                
            except Exception as e:
                st.warning(f"⚠️ Failed to load model {model_name}: {e}")
        
        progress_bar.empty()
        status_text.empty()
        
        if self.models:
            st.success(f"✅ Successfully loaded {len(self.models)} models")
            
            # Display model information
            model_info_data = []
            for name, info in self.models.items():
                accuracy = info.get('accuracy', 'unknown')
                acc_str = f"{accuracy:.4f}" if isinstance(accuracy, (int, float)) else str(accuracy)
                model_info_data.append({
                    'Model Name': name,
                    'Accuracy': acc_str,
                    'Vocab Size': info['model_params'].get('vocab_size', 'unknown'),
                    'Max Length': info['model_params'].get('max_length', 'unknown')
                })
            
            st.write("📊 **Available Models:**")
            st.dataframe(pd.DataFrame(model_info_data), use_container_width=True)
        else:
            st.error("❌ No models could be loaded successfully")
    
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
        try:
            from myword import MyWord
            self.myword_tokenizer = MyWord()
            st.success("✅ MyWord tokenizer initialized successfully!")
        except Exception as e:
            st.warning(f"⚠️ Error initializing MyWord tokenizer: {e}")
            st.write("📝 Using simple word splitting as fallback")
            self.myword_tokenizer = None
    
    def clean_text(self, text):
        """Basic text cleaning similar to training pipeline"""
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
        """Tokenize Myanmar text using myWord or fallback"""
        if not self.myword_tokenizer:
            # Fallback to simple splitting
            return text.split()
        
        try:
            # Use myWord tokenizer if available (check for correct method name)
            if hasattr(self.myword_tokenizer, 'tokenize'):
                tokens = self.myword_tokenizer.tokenize(text)
            elif hasattr(self.myword_tokenizer, 'word_tokenize'):
                tokens = self.myword_tokenizer.word_tokenize(text)
            else:
                # Fallback
                tokens = text.split()
            return tokens
        except Exception as e:
            # Fallback to simple splitting
            return text.split()
    
    def predict_with_all_models(self, text):
        """Predict article category using all available models"""
        if not self.models:
            return {}
        
        # Clean and tokenize text once
        cleaned_text = self.clean_text(text)
        tokens = self.tokenize_text(cleaned_text)
        tokens_text = ' '.join(tokens)
        
        results = {}
        
        for model_name, model_info in self.models.items():
            try:
                model = model_info['model']
                tokenizer = model_info['tokenizer']
                model_params = model_info['model_params']
                
                # Convert to sequence
                sequence = tokenizer.texts_to_sequences([tokens_text])
                
                # Pad sequence
                padded_sequence = pad_sequences(
                    sequence, 
                    maxlen=model_params['max_length'], 
                    padding='post', 
                    truncating='post'
                )
                
                # Predict
                prediction_probs = model.predict(padded_sequence, verbose=0)[0]
                predicted_class = np.argmax(prediction_probs)
                
                # Get class name
                label_mapping = model_params.get('label_mapping', {0: 'red', 1: 'neutral', 2: 'green'})
                predicted_label = label_mapping[predicted_class]
                
                results[model_name] = {
                    'predicted_class': int(predicted_class),
                    'predicted_label': predicted_label,
                    'probabilities': {
                        'red': float(prediction_probs[0]),
                        'neutral': float(prediction_probs[1]),
                        'green': float(prediction_probs[2])
                    },
                    'confidence': float(np.max(prediction_probs)),
                    'model_accuracy': model_info.get('accuracy', 'unknown')
                }
                
            except Exception as e:
                results[model_name] = {'error': str(e)}
        
        return results, len(tokens)

def main():
    st.title("🇲🇲 Myanmar Article Multi-Model Classifier")
    st.write("Compare predictions from multiple trained BiLSTM models")
    
    # Initialize classifier
    if 'classifier' not in st.session_state:
        with st.spinner("Loading models..."):
            st.session_state.classifier = MultiModelClassifier()
    
    classifier = st.session_state.classifier
    
    if not classifier.models:
        st.error("❌ No models available. Please train some models first.")
        return
    
    # Sidebar for model selection
    st.sidebar.header("🔧 Model Controls")
    
    # Model selection
    selected_models = st.sidebar.multiselect(
        "Select models to compare:",
        options=list(classifier.models.keys()),
        default=list(classifier.models.keys())[:3],  # Select first 3 by default
        help="Choose which models to use for comparison"
    )
    
    # Input methods
    st.header("📝 Input Article")
    
    input_method = st.radio(
        "Choose input method:",
        ["Text Input", "File Upload"],
        horizontal=True
    )
    
    article_text = ""
    
    if input_method == "Text Input":
        article_text = st.text_area(
            "Enter Myanmar article text:",
            height=200,
            placeholder="မြန်မာ ဆောင်းပါး ရေးပါ..."
        )
    else:
        uploaded_file = st.file_uploader(
            "Upload text file:",
            type=['txt'],
            help="Upload a .txt file containing Myanmar text"
        )
        
        if uploaded_file is not None:
            article_text = str(uploaded_file.read(), "utf-8")
            st.text_area("File content:", value=article_text, height=100)
    
    # Analysis
    if st.button("🔍 Analyze Article", type="primary", disabled=not article_text.strip()):
        if not selected_models:
            st.warning("⚠️ Please select at least one model for comparison")
            return
            
        with st.spinner("Analyzing with all selected models..."):
            # Filter classifier models to only selected ones
            original_models = classifier.models.copy()
            classifier.models = {k: v for k, v in original_models.items() if k in selected_models}
            
            # Get predictions
            predictions, token_count = classifier.predict_with_all_models(article_text)
            
            # Restore original models
            classifier.models = original_models
        
        if predictions:
            st.header("📊 Multi-Model Analysis Results")
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Models Used", len(predictions))
            with col2:
                st.metric("Token Count", token_count)
            with col3:
                successful_predictions = len([p for p in predictions.values() if 'error' not in p])
                st.metric("Successful Predictions", successful_predictions)
            
            # Detailed results
            st.subheader("🔍 Individual Model Predictions")
            
            # Create comparison table
            comparison_data = []
            for model_name, result in predictions.items():
                if 'error' not in result:
                    accuracy = result.get('model_accuracy', 'unknown')
                    acc_str = f"{accuracy:.4f}" if isinstance(accuracy, (int, float)) else str(accuracy)
                    
                    comparison_data.append({
                        'Model': model_name,
                        'Prediction': result['predicted_label'].upper(),
                        'Confidence': f"{result['confidence']:.3f}",
                        'Training Accuracy': acc_str,
                        'Red Prob': f"{result['probabilities']['red']:.3f}",
                        'Neutral Prob': f"{result['probabilities']['neutral']:.3f}",
                        'Green Prob': f"{result['probabilities']['green']:.3f}"
                    })
                else:
                    comparison_data.append({
                        'Model': model_name,
                        'Prediction': 'ERROR',
                        'Confidence': '-',
                        'Training Accuracy': '-',
                        'Red Prob': '-',
                        'Neutral Prob': '-',
                        'Green Prob': '-'
                    })
            
            # Display comparison table
            df = pd.DataFrame(comparison_data)
            
            # Color-code predictions with darker backgrounds for better contrast
            def color_prediction(val):
                if val == 'RED':
                    return 'background-color: #ff6b6b; color: white; font-weight: bold'
                elif val == 'GREEN':
                    return 'background-color: #4ecdc4; color: white; font-weight: bold'
                elif val == 'NEUTRAL':
                    return 'background-color: #ffa726; color: white; font-weight: bold'
                return ''
            
            styled_df = df.style.applymap(color_prediction, subset=['Prediction'])
            st.dataframe(styled_df, use_container_width=True)
            
            # Consensus analysis
            st.subheader("🤝 Model Consensus Analysis")
            
            successful_results = {k: v for k, v in predictions.items() if 'error' not in v}
            if len(successful_results) > 1:
                # Count predictions
                prediction_counts = {}
                for result in successful_results.values():
                    pred = result['predicted_label']
                    prediction_counts[pred] = prediction_counts.get(pred, 0) + 1
                
                # Consensus
                most_common = max(prediction_counts, key=prediction_counts.get)
                consensus_strength = prediction_counts[most_common] / len(successful_results)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Consensus Prediction", most_common.upper())
                with col2:
                    st.metric("Agreement Level", f"{consensus_strength:.1%}")
                
                # Agreement breakdown
                st.write("**Prediction Distribution:**")
                for pred, count in sorted(prediction_counts.items()):
                    percentage = count / len(successful_results) * 100
                    st.write(f"- {pred.upper()}: {count}/{len(successful_results)} models ({percentage:.1f}%)")
            
            # Detailed model cards
            st.subheader("📋 Detailed Model Results")
            
            for model_name, result in predictions.items():
                with st.expander(f"🤖 {model_name}"):
                    if 'error' not in result:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Prediction:** {result['predicted_label'].upper()}")
                            st.write(f"**Confidence:** {result['confidence']:.3f}")
                            accuracy = result.get('model_accuracy', 'unknown')
                            acc_str = f"{accuracy:.4f}" if isinstance(accuracy, (int, float)) else str(accuracy)
                            st.write(f"**Training Accuracy:** {acc_str}")
                        
                        with col2:
                            st.write("**Class Probabilities:**")
                            probs = result['probabilities']
                            st.write(f"- RED: {probs['red']:.3f}")
                            st.write(f"- NEUTRAL: {probs['neutral']:.3f}")
                            st.write(f"- GREEN: {probs['green']:.3f}")
                        
                        # Probability bar chart
                        prob_data = pd.DataFrame([
                            {'Class': 'RED', 'Probability': probs['red']},
                            {'Class': 'NEUTRAL', 'Probability': probs['neutral']},
                            {'Class': 'GREEN', 'Probability': probs['green']}
                        ])
                        st.bar_chart(prob_data.set_index('Class')['Probability'])
                        
                    else:
                        st.error(f"❌ Error: {result['error']}")
        else:
            st.error("❌ No predictions could be generated")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "💡 **Tip:** Different models may have varying performance based on their training data and architecture. "
        "Compare results across models to get a more comprehensive understanding."
    )

if __name__ == "__main__":
    main()