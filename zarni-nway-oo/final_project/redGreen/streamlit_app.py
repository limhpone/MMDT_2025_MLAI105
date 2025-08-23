#!/usr/bin/env python3
"""
Myanmar Article Classification - Streamlit Web App
Real-time article classification using Bi-LSTM model
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

# Add the myWord tokenizer path
sys.path.append(os.path.join(os.path.dirname(__file__), '2_processor', 'tokenizer', 'myWord'))

# Page configuration
st.set_page_config(
    page_title="Myanmar Article Classifier",
    page_icon="🇲🇲",
    layout="wide",
    initial_sidebar_state="expanded"
)

class MyanmarArticleClassifier:
    def __init__(self, model_dir):
        """Initialize the classifier with model artifacts"""
        self.model_dir = model_dir
        self.model = None
        self.tokenizer = None
        self.myword_tokenizer = None
        self.model_params = None
        
        # Load model and artifacts
        self._load_model_artifacts()
        
        # Initialize Myanmar tokenizer
        self._initialize_myword()
    
    def _load_model_artifacts(self):
        """Load trained model and preprocessing artifacts"""
        try:
            # Load model
            model_path = os.path.join(self.model_dir, 'bilstm_model.h5')
            self.model = load_model(model_path)
            
            # Load tokenizer
            tokenizer_path = os.path.join(self.model_dir, 'tokenizer.pickle')
            with open(tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
            
            # Load model parameters
            params_path = os.path.join(self.model_dir, 'model_params.pickle')
            with open(params_path, 'rb') as f:
                self.model_params = pickle.load(f)
            
            st.success("✅ Model loaded successfully!")
            
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            st.stop()
    
    def _initialize_myword(self):
        """Initialize myWord tokenizer"""
        try:
            from myword import MyWord
            self.myword_tokenizer = MyWord()
            
            if hasattr(self.myword_tokenizer, 'initialized') and self.myword_tokenizer.initialized:
                st.success("✅ MyWord tokenizer initialized successfully!")
            else:
                st.warning("⚠️ MyWord tokenizer initialized but may have issues")
                
        except Exception as e:
            st.error(f"❌ Error initializing MyWord tokenizer: {e}")
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
        """Tokenize text using myWord"""
        if not self.myword_tokenizer:
            st.warning("Using fallback tokenization (whitespace split)")
            return text.split()
        
        try:
            tokens = self.myword_tokenizer.segment(text)
            return tokens
        except Exception as e:
            st.error(f"Error in myWord tokenization: {e}")
            return text.split()
    
    def preprocess_article(self, title, content):
        """Preprocess article text for model input"""
        # Combine title and content
        full_text = f"{title}\n\n{content}" if title.strip() else content
        
        # Apply myword tokenization to ensure consistency with training data
        cleaned_text = self.clean_text(full_text)
        tokens = self.tokenize_text(cleaned_text)
        tokens_text = ' '.join(tokens)
        
        # Convert to sequence using training tokenizer
        sequence = self.tokenizer.texts_to_sequences([tokens_text])
        
        # Pad sequence
        padded_sequence = pad_sequences(
            sequence, 
            maxlen=self.model_params['max_length'], 
            padding='post', 
            truncating='post'
        )
        
        return padded_sequence, tokens, tokens_text, cleaned_text
    
    def predict_article(self, title, content):
        """Predict article category and return detailed results"""
        try:
            # Preprocess
            padded_sequence, tokens, tokens_text, cleaned_text = self.preprocess_article(title, content)
            
            # Predict
            prediction_probs = self.model.predict(padded_sequence, verbose=0)[0]
            predicted_class = np.argmax(prediction_probs)
            predicted_label = self.model_params['label_mapping'][predicted_class]
            confidence = float(prediction_probs[predicted_class])
            
            # Calculate additional metrics
            probabilities = {
                self.model_params['label_mapping'][i]: float(prob) 
                for i, prob in enumerate(prediction_probs)
            }
            
            return {
                'success': True,
                'predicted_class': int(predicted_class),
                'predicted_label': predicted_label,
                'confidence': confidence,
                'probabilities': probabilities,
                'tokens': tokens,
                'tokens_text': tokens_text,
                'cleaned_text': cleaned_text,
                'token_count': len(tokens),
                'text_length': len(cleaned_text),
                'sequence_length': len(padded_sequence[0][padded_sequence[0] > 0])  # Non-zero elements
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

@st.cache_resource
def load_classifier():
    """Load the classifier (cached for performance)"""
    # Use the final model directory from utils
    from utils import get_data_directories
    dirs = get_data_directories()
    model_dir = dirs['final_model']
    return MyanmarArticleClassifier(model_dir)

def display_prediction_results(result):
    """Display prediction results in a nice format"""
    if not result['success']:
        st.error(f"❌ Prediction failed: {result['error']}")
        return
    
    # Main prediction result
    st.markdown("## 🎯 Prediction Results")
    
    # Create columns for main metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Predicted category with color coding
        label = result['predicted_label'].upper()
        if label == 'RED':
            st.markdown(f"### 🔴 **{label}**")
        elif label == 'GREEN':
            st.markdown(f"### 🟢 **{label}**")
        else:  # NEUTRAL
            st.markdown(f"### ⚪ **{label}**")
    
    with col2:
        confidence_pct = result['confidence'] * 100
        st.metric("Confidence", f"{confidence_pct:.1f}%")
    
    with col3:
        # Risk assessment based on confidence
        if result['confidence'] >= 0.9:
            st.markdown("### ✅ **HIGH CONFIDENCE**")
        elif result['confidence'] >= 0.7:
            st.markdown("### ⚠️ **MEDIUM CONFIDENCE**")
        else:
            st.markdown("### 🔴 **LOW CONFIDENCE**")
    
    # Detailed probabilities
    st.markdown("## 📊 Category Probabilities")
    
    prob_data = []
    for category, prob in result['probabilities'].items():
        prob_data.append({
            'Category': category.upper(),
            'Probability': f"{prob:.3f}",
            'Percentage': f"{prob*100:.1f}%"
        })
    
    # Sort by probability
    prob_data = sorted(prob_data, key=lambda x: float(x['Probability']), reverse=True)
    prob_df = pd.DataFrame(prob_data)
    
    # Display as a nice table
    st.dataframe(prob_df, use_container_width=True, hide_index=True)
    
    # Probability visualization
    categories = [item['Category'] for item in prob_data]
    probabilities = [float(item['Probability']) for item in prob_data]
    
    # Create a bar chart
    chart_data = pd.DataFrame({
        'Category': categories,
        'Probability': probabilities
    })
    
    st.bar_chart(chart_data.set_index('Category'))
    
    # Technical details in expander
    with st.expander("🔧 Technical Details"):
        tech_col1, tech_col2 = st.columns(2)
        
        with tech_col1:
            st.metric("Total Tokens", result['token_count'])
            st.metric("Text Length", f"{result['text_length']} chars")
            st.metric("Sequence Length", result['sequence_length'])
        
        with tech_col2:
            st.write("**Model Info:**")
            st.write("- Architecture: Bidirectional LSTM")
            st.write("- Tokenization: Myanmar myWord")
            st.write("- Categories: Red, Neutral, Green")
        
        # Show tokenized text sample
        st.write("**Tokenized Text Sample:**")
        sample_tokens = ' '.join(result['tokens'][:50])
        if len(result['tokens']) > 50:
            sample_tokens += "..."
        st.code(sample_tokens, language="text")

def main():
    """Main Streamlit app"""
    # App header
    st.title("🇲🇲 Myanmar Article Classification")
    st.markdown("**Real-time news article classification using Bi-LSTM deep learning model**")
    st.markdown("---")
    
    # Sidebar info
    with st.sidebar:
        st.header("📋 About")
        st.write("""
        This app classifies Myanmar news articles into three categories:
        - 🔴 **Red**: Pro-democracy/opposition content
        - 🟢 **Green**: Government/military favorable content
        - ⚪ **Neutral**: Balanced/neutral content
        """)
        
        st.header("🛠️ Model Info")
        st.write("""
        - **Architecture**: Bidirectional LSTM
        - **Tokenization**: Myanmar myWord segmentation
        - **Training Data**: 1,330+ labeled articles
        - **Accuracy**: 95%+ on test data
        """)
        
        st.header("📊 Usage Statistics")
        if 'prediction_count' not in st.session_state:
            st.session_state.prediction_count = 0
        st.metric("Predictions Made", st.session_state.prediction_count)
    
    # Load classifier
    with st.spinner("Loading model..."):
        classifier = load_classifier()
    
    # Input form
    st.header("📝 Article Input")
    
    with st.form("article_form"):
        # Title input
        title = st.text_input(
            "Article Title (Optional)",
            placeholder="Enter the article title here...",
            help="The title of the news article (optional)"
        )
        
        # Content input
        content = st.text_area(
            "Article Content *",
            placeholder="Paste the full article content here in Myanmar language...",
            height=200,
            help="The main body content of the article (required)"
        )
        
        # Predict button
        predict_button = st.form_submit_button(
            "🎯 Predict Category",
            type="primary",
            use_container_width=True
        )
    
    # Handle prediction
    if predict_button:
        if not content.strip():
            st.error("❌ Please enter article content to proceed with prediction.")
        else:
            # Show prediction
            with st.spinner("🔮 Analyzing article..."):
                result = classifier.predict_article(title, content)
                
                # Update prediction counter
                st.session_state.prediction_count += 1
                
                # Display results
                display_prediction_results(result)
                
                # Show timestamp
                st.markdown(f"*Prediction completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Myanmar Article Classifier** | Built with ❤️ using Streamlit and TensorFlow | "
        "🤖 Powered by Bi-LSTM Neural Network"
    )

if __name__ == "__main__":
    main()