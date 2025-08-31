#!/usr/bin/env python3
"""
BiLSTM + CRF Model for Sequence Labeling
Uses CRF layer for better sequence prediction
"""

import os
import sys
import pickle
import numpy as np
import tensorflow as tf
from datetime import datetime
import json
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow_addons as tfa

class BiLSTMCRF:
    def __init__(self):
        self.model = None
        
    def create_model(self, vocab_size, max_length, num_classes):
        """Create BiLSTM + CRF model"""
        # Input layer
        input_layer = Input(shape=(max_length,))
        
        # Embedding layer
        embedding = Embedding(vocab_size, 128, mask_zero=True)(input_layer)
        embedding = Dropout(0.2)(embedding)
        
        # BiLSTM layers
        lstm1 = Bidirectional(LSTM(64, return_sequences=True, dropout=0.3))(embedding)
        lstm2 = Bidirectional(LSTM(32, return_sequences=True, dropout=0.3))(lstm1)
        
        # Dense layer before CRF
        dense = Dense(num_classes)(lstm2)
        
        # CRF layer
        crf_layer = tfa.layers.CRF(num_classes)
        output = crf_layer(dense)
        
        model = Model(inputs=input_layer, outputs=output)
        
        # Compile with CRF loss
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=crf_layer.loss,
            metrics=[crf_layer.accuracy]
        )
        
        return model

def main():
    """Main training function for BiLSTM + CRF"""
    print("🤖 BiLSTM + CRF Training Started")
    
    # Load cached data
    cache_path = os.path.join('..', 'data_cache.pkl')
    with open(cache_path, 'rb') as f:
        data = pickle.load(f)
    
    texts = data['texts']
    labels = data['labels']
    data_info = data['data_info']
    
    # Convert to sequences (similar to enhanced_bilstm)
    word_to_idx = data_info['word_to_idx']
    max_length = min(data_info['max_length'], 500)
    
    sequences = []
    for text in texts:
        tokens = text.split()
        sequence = [word_to_idx.get(token, 1) for token in tokens]
        sequences.append(sequence)
    
    X = pad_sequences(sequences, maxlen=max_length, padding='post')
    y = labels
    
    # For CRF, we need to convert labels to sequence format
    y_sequences = []
    for label in y:
        # Create a sequence where each token gets the same label
        seq_labels = [label] * max_length
        y_sequences.append(seq_labels)
    y_sequences = np.array(y_sequences)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_sequences, test_size=0.2, random_state=42)
    
    # Create and train model
    trainer = BiLSTMCRF()
    model = trainer.create_model(data_info['vocab_size'], max_length, data_info['num_classes'])
    
    print("🚀 Training BiLSTM + CRF model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=30,
        batch_size=32,
        verbose=1
    )
    
    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\n📊 Final Results:")
    print(f"   Test Accuracy: {test_acc:.4f}")
    print(f"   Test Loss: {test_loss:.4f}")
    
    # Save results
    results = {
        'model_name': 'bilstm_crf',
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss)
    }
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(f'bilstm_crf_results_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ BiLSTM + CRF training completed!")

if __name__ == "__main__":
    main()
