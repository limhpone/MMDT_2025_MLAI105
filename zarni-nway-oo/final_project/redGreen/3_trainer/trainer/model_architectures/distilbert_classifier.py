#!/usr/bin/env python3
"""
DistilBERT Classifier
Uses pre-trained DistilBERT for Myanmar text classification
"""

import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import json
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
from transformers import InputExample, InputFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class DistilBERTClassifier:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        
    def prepare_data(self, texts, labels, max_length=512):
        """Prepare data for DistilBERT"""
        # Initialize tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
        
        # Tokenize texts
        input_ids = []
        attention_masks = []
        
        for text in texts:
            # Join tokens back to text (since our data is pre-tokenized)
            text_str = ' '.join(text.split()[:max_length-2])  # Reserve space for [CLS] and [SEP]
            
            encoded = self.tokenizer.encode_plus(
                text_str,
                add_special_tokens=True,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='tf'
            )
            
            input_ids.append(encoded['input_ids'].numpy()[0])
            attention_masks.append(encoded['attention_mask'].numpy()[0])
        
        return np.array(input_ids), np.array(attention_masks)
    
    def create_model(self, num_classes):
        """Create DistilBERT model"""
        self.model = TFDistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-multilingual-cased',
            num_labels=num_classes
        )
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
        self.model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        
        return self.model

def main():
    """Main training function for DistilBERT"""
    print("🤖 DistilBERT Training Started")
    
    try:
        # Load cached data
        cache_path = os.path.join('..', 'data_cache.pkl')
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        
        texts = data['texts']
        labels = data['labels']
        data_info = data['data_info']
        
        # Initialize classifier
        classifier = DistilBERTClassifier()
        
        print("🔄 Preparing data for DistilBERT...")
        input_ids, attention_masks = classifier.prepare_data(texts, labels)
        
        # Split data
        X_train_ids, X_test_ids, X_train_masks, X_test_masks, y_train, y_test = train_test_split(
            input_ids, attention_masks, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Create model
        model = classifier.create_model(data_info['num_classes'])
        
        print("🚀 Training DistilBERT model...")
        history = model.fit(
            [X_train_ids, X_train_masks], y_train,
            validation_data=([X_test_ids, X_test_masks], y_test),
            epochs=3,  # Few epochs for BERT
            batch_size=16,  # Small batch size
            verbose=1
        )
        
        # Evaluate
        test_loss, test_acc = model.evaluate([X_test_ids, X_test_masks], y_test, verbose=0)
        
        # Get predictions
        predictions = model.predict([X_test_ids, X_test_masks])
        y_pred_classes = np.argmax(predictions.logits, axis=1)
        
        print(f"\n📊 Final Results:")
        print(f"   Test Accuracy: {test_acc:.4f}")
        print(f"   Test Loss: {test_loss:.4f}")
        
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, y_pred_classes, target_names=['red', 'neutral', 'green']))
        
        # Save results
        results = {
            'model_name': 'distilbert_classifier',
            'test_accuracy': float(test_acc),
            'test_loss': float(test_loss)
        }
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(f'distilbert_results_{timestamp}.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ DistilBERT training completed!")
        
    except Exception as e:
        print(f"❌ DistilBERT training failed: {e}")
        print("Note: DistilBERT requires transformers library. Install with: pip install transformers")
        
        # Save error result
        results = {
            'model_name': 'distilbert_classifier',
            'error': str(e),
            'status': 'failed'
        }
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(f'distilbert_results_{timestamp}.json', 'w') as f:
            json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
