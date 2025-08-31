#!/usr/bin/env python3
"""
Enhanced BiLSTM Model with Attention, Regularization and Hyperparameter Tuning
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import json
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Embedding, LSTM, Bidirectional, Dense, 
                                     Dropout, BatchNormalization, Attention, 
                                     GlobalMaxPooling1D, Concatenate)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import optuna

class EnhancedBiLSTM:
    def __init__(self):
        self.model = None
        self.history = None
        self.best_params = None
        
    def create_model(self, vocab_size, max_length, num_classes, **params):
        """Create enhanced BiLSTM model with attention"""
        # Input layer
        input_layer = Input(shape=(max_length,))
        
        # Embedding layer
        embedding = Embedding(
            input_dim=vocab_size,
            output_dim=params.get('embedding_dim', 128),
            input_length=max_length,
            mask_zero=True
        )(input_layer)
        
        # Dropout after embedding
        x = Dropout(params.get('embedding_dropout', 0.2))(embedding)
        
        # First BiLSTM layer
        lstm1 = Bidirectional(LSTM(
            params.get('lstm1_units', 64),
            return_sequences=True,
            dropout=params.get('lstm_dropout', 0.3),
            recurrent_dropout=params.get('recurrent_dropout', 0.2)
        ))(x)
        
        # Batch normalization
        lstm1 = BatchNormalization()(lstm1)
        
        # Second BiLSTM layer (optional)
        if params.get('use_second_lstm', True):
            lstm2 = Bidirectional(LSTM(
                params.get('lstm2_units', 32),
                return_sequences=True,
                dropout=params.get('lstm_dropout', 0.3),
                recurrent_dropout=params.get('recurrent_dropout', 0.2)
            ))(lstm1)
            lstm2 = BatchNormalization()(lstm2)
            lstm_output = lstm2
        else:
            lstm_output = lstm1
        
        # Attention mechanism
        if params.get('use_attention', True):
            attention = tf.keras.layers.MultiHeadAttention(
                num_heads=params.get('attention_heads', 4),
                key_dim=params.get('attention_dim', 32)
            )(lstm_output, lstm_output)
            
            # Combine LSTM and attention outputs
            combined = Concatenate()([lstm_output, attention])
        else:
            combined = lstm_output
        
        # Global pooling
        pooled = GlobalMaxPooling1D()(combined)
        
        # Dense layers
        dense1 = Dense(
            params.get('dense1_units', 64),
            activation='relu'
        )(pooled)
        dense1 = Dropout(params.get('dense_dropout', 0.4))(dense1)
        dense1 = BatchNormalization()(dense1)
        
        if params.get('use_second_dense', True):
            dense2 = Dense(
                params.get('dense2_units', 32),
                activation='relu'
            )(dense1)
            dense2 = Dropout(params.get('dense_dropout', 0.4))(dense2)
            dense_output = dense2
        else:
            dense_output = dense1
        
        # Output layer
        output = Dense(num_classes, activation='softmax')(dense_output)
        
        model = Model(inputs=input_layer, outputs=output)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=params.get('learning_rate', 0.001)),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def objective(self, trial, X_train, y_train, X_val, y_val, data_info):
        """Optuna objective function for hyperparameter optimization"""
        # Hyperparameters to optimize
        params = {
            'embedding_dim': trial.suggest_categorical('embedding_dim', [64, 128, 256]),
            'lstm1_units': trial.suggest_categorical('lstm1_units', [32, 64, 128]),
            'lstm2_units': trial.suggest_categorical('lstm2_units', [16, 32, 64]),
            'embedding_dropout': trial.suggest_float('embedding_dropout', 0.1, 0.4),
            'lstm_dropout': trial.suggest_float('lstm_dropout', 0.2, 0.5),
            'recurrent_dropout': trial.suggest_float('recurrent_dropout', 0.1, 0.3),
            'dense_dropout': trial.suggest_float('dense_dropout', 0.3, 0.6),
            'dense1_units': trial.suggest_categorical('dense1_units', [32, 64, 128]),
            'dense2_units': trial.suggest_categorical('dense2_units', [16, 32, 64]),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'use_second_lstm': trial.suggest_categorical('use_second_lstm', [True, False]),
            'use_attention': trial.suggest_categorical('use_attention', [True, False]),
            'use_second_dense': trial.suggest_categorical('use_second_dense', [True, False]),
            'attention_heads': trial.suggest_categorical('attention_heads', [2, 4, 8]),
            'attention_dim': trial.suggest_categorical('attention_dim', [16, 32, 64])
        }
        
        try:
            # Create model
            model = self.create_model(
                data_info['vocab_size'],
                min(data_info['max_length'], 500),  # Cap max length
                data_info['num_classes'],
                **params
            )
            
            # Callbacks
            callbacks = [
                EarlyStopping(patience=5, restore_best_weights=True),
                ReduceLROnPlateau(patience=3, factor=0.5)
            ]
            
            # Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=20,  # Reduced for optimization
                batch_size=32,
                callbacks=callbacks,
                verbose=0
            )
            
            # Return validation accuracy
            return max(history.history['val_accuracy'])
            
        except Exception as e:
            print(f"Trial failed: {e}")
            return 0.0
    
    def hyperparameter_optimization(self, X_train, y_train, X_val, y_val, data_info, n_trials=20):
        """Run hyperparameter optimization"""
        print("🔍 Starting hyperparameter optimization...")
        
        study = optuna.create_study(direction='maximize')
        study.optimize(
            lambda trial: self.objective(trial, X_train, y_train, X_val, y_val, data_info),
            n_trials=n_trials
        )
        
        self.best_params = study.best_params
        print(f"✅ Best parameters found: {self.best_params}")
        print(f"   Best validation accuracy: {study.best_value:.4f}")
        
        return study.best_params, study.best_value
    
    def train_final_model(self, X_train, y_train, X_val, y_val, data_info, class_weights=None):
        """Train the final model with best parameters"""
        print("🚀 Training final model with optimized parameters...")
        
        # Create model with best parameters
        self.model = self.create_model(
            data_info['vocab_size'],
            min(data_info['max_length'], 500),
            data_info['num_classes'],
            **self.best_params
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-7),
            ModelCheckpoint(
                'best_enhanced_bilstm.h5',
                save_best_only=True,
                monitor='val_accuracy'
            )
        ]
        
        # Train final model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=32,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.model, self.history

def main():
    """Main training function"""
    print("🤖 Enhanced BiLSTM Training Started")
    
    # Load cached data
    cache_path = os.path.join(os.path.dirname(__file__), '..', 'data_cache.pkl')
    with open(cache_path, 'rb') as f:
        data = pickle.load(f)
    
    texts = data['texts']
    labels = data['labels']
    data_info = data['data_info']
    
    # Convert texts to sequences
    word_to_idx = data_info['word_to_idx']
    max_length = min(data_info['max_length'], 500)  # Cap at 500
    
    sequences = []
    for text in texts:
        tokens = text.split()
        sequence = [word_to_idx.get(token, 1) for token in tokens]  # 1 is OOV
        sequences.append(sequence)
    
    X = pad_sequences(sequences, maxlen=max_length, padding='post')
    y = labels
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    # Calculate class weights
    class_weights_array = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = {i: class_weights_array[i] for i in range(len(class_weights_array))}
    
    print(f"📊 Data splits: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # Initialize trainer
    trainer = EnhancedBiLSTM()
    
    # Hyperparameter optimization
    best_params, best_score = trainer.hyperparameter_optimization(
        X_train, y_train, X_val, y_val, data_info, n_trials=30
    )
    
    # Train final model
    model, history = trainer.train_final_model(
        X_train, y_train, X_val, y_val, data_info, class_weights
    )
    
    # Evaluate on test set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    print(f"\n📊 Final Results:")
    print(f"   Test Accuracy: {test_acc:.4f}")
    print(f"   Test Loss: {test_loss:.4f}")
    
    # Classification report
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred_classes, 
                              target_names=['red', 'neutral', 'green']))
    
    # Save results
    results = {
        'model_name': 'enhanced_bilstm',
        'best_params': best_params,
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'optimization_score': float(best_score)
    }
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(f'enhanced_bilstm_results_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Enhanced BiLSTM training completed!")
    print(f"📁 Results saved: enhanced_bilstm_results_{timestamp}.json")

if __name__ == "__main__":
    main()
