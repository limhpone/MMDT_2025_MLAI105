#!/usr/bin/env python3
"""
CNN + BiLSTM Hybrid Model
Uses CNN for local feature extraction and BiLSTM for sequence modeling
"""

import os
import pickle
import numpy as np
import tensorflow as tf
from datetime import datetime
import json
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Embedding, Conv1D, MaxPooling1D, 
                                     LSTM, Bidirectional, Dense, Dropout, 
                                     GlobalMaxPooling1D, Concatenate)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class CNNBiLSTMHybrid:
    def __init__(self):
        self.model = None
        
    def create_model(self, vocab_size, max_length, num_classes):
        """Create CNN + BiLSTM hybrid model"""
        # Input layer
        input_layer = Input(shape=(max_length,))
        
        # Embedding layer
        embedding = Embedding(vocab_size, 128, input_length=max_length)(input_layer)
        embedding = Dropout(0.2)(embedding)
        
        # CNN branch - multiple filter sizes
        cnn_outputs = []
        filter_sizes = [2, 3, 4, 5]
        
        for filter_size in filter_sizes:
            conv = Conv1D(filters=64, kernel_size=filter_size, activation='relu')(embedding)
            pool = MaxPooling1D(pool_size=2)(conv)
            cnn_outputs.append(pool)
        
        # Concatenate CNN outputs
        cnn_concat = Concatenate()(cnn_outputs)
        cnn_features = GlobalMaxPooling1D()(cnn_concat)
        
        # BiLSTM branch
        lstm = Bidirectional(LSTM(64, return_sequences=True, dropout=0.3))(embedding)
        lstm = Bidirectional(LSTM(32, dropout=0.3))(lstm)
        
        # Combine CNN and LSTM features
        combined = Concatenate()([cnn_features, lstm])
        
        # Dense layers
        dense1 = Dense(128, activation='relu')(combined)
        dense1 = Dropout(0.4)(dense1)
        
        dense2 = Dense(64, activation='relu')(dense1)
        dense2 = Dropout(0.4)(dense2)
        
        # Output layer
        output = Dense(num_classes, activation='softmax')(dense2)
        
        model = Model(inputs=input_layer, outputs=output)
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

def main():
    """Main training function for CNN + BiLSTM"""
    print("🤖 CNN + BiLSTM Hybrid Training Started")
    
    # Load cached data
    cache_path = os.path.join('..', 'data_cache.pkl')
    with open(cache_path, 'rb') as f:
        data = pickle.load(f)
    
    texts = data['texts']
    labels = data['labels']
    data_info = data['data_info']
    
    # Convert to sequences
    word_to_idx = data_info['word_to_idx']
    max_length = min(data_info['max_length'], 500)
    
    sequences = []
    for text in texts:
        tokens = text.split()
        sequence = [word_to_idx.get(token, 1) for token in tokens]
        sequences.append(sequence)
    
    X = pad_sequences(sequences, maxlen=max_length, padding='post')
    y = labels
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Create and train model
    trainer = CNNBiLSTMHybrid()
    model = trainer.create_model(data_info['vocab_size'], max_length, data_info['num_classes'])
    
    print("🚀 Training CNN + BiLSTM model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=30,
        batch_size=32,
        verbose=1
    )
    
    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    print(f"\n📊 Final Results:")
    print(f"   Test Accuracy: {test_acc:.4f}")
    print(f"   Test Loss: {test_loss:.4f}")
    
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=['red', 'neutral', 'green']))
    
    # Save results
    results = {
        'model_name': 'cnn_bilstm_hybrid',
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss)
    }
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(f'cnn_bilstm_results_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ CNN + BiLSTM training completed!")

if __name__ == "__main__":
    main()
