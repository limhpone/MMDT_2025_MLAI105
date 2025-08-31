#!/usr/bin/env python3
"""
Lightweight Transformer Encoder Model
Custom transformer implementation for text classification
"""

import os
import pickle
import numpy as np
import tensorflow as tf
from datetime import datetime
import json
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Embedding, Dense, Dropout, LayerNormalization,
                                     MultiHeadAttention, GlobalAveragePooling1D)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"), 
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

class LightweightTransformer:
    def __init__(self):
        self.model = None
        
    def create_model(self, vocab_size, max_length, num_classes):
        """Create lightweight transformer model"""
        embed_dim = 128  # Embedding size for each token
        num_heads = 8  # Number of attention heads
        ff_dim = 256  # Hidden layer size in feed forward network
        
        inputs = Input(shape=(max_length,))
        embedding_layer = TokenAndPositionEmbedding(max_length, vocab_size, embed_dim)
        x = embedding_layer(inputs)
        
        # Two transformer blocks
        transformer_block1 = TransformerBlock(embed_dim, num_heads, ff_dim)
        x = transformer_block1(x, training=True)
        
        transformer_block2 = TransformerBlock(embed_dim, num_heads, ff_dim)
        x = transformer_block2(x)
        
        # Global average pooling
        x = GlobalAveragePooling1D()(x)
        x = Dropout(0.1)(x)
        
        # Dense layers
        x = Dense(64, activation="relu")(x)
        x = Dropout(0.1)(x)
        
        outputs = Dense(num_classes, activation="softmax")(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        return model

def main():
    """Main training function for Transformer"""
    print("🤖 Lightweight Transformer Training Started")
    
    # Load cached data
    cache_path = os.path.join(os.path.dirname(__file__), '..', 'data_cache.pkl')
    with open(cache_path, 'rb') as f:
        data = pickle.load(f)
    
    texts = data['texts']
    labels = data['labels']
    data_info = data['data_info']
    
    # Convert to sequences
    word_to_idx = data_info['word_to_idx']
    max_length = min(data_info['max_length'], 400)  # Cap for transformer
    
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
    transformer = LightweightTransformer()
    model = transformer.create_model(data_info['vocab_size'], max_length, data_info['num_classes'])
    
    print(f"Model summary:")
    model.summary()
    
    print("🚀 Training Transformer model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=25,
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
        'model_name': 'transformer_encoder',
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss)
    }
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(f'transformer_results_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Transformer training completed!")

if __name__ == "__main__":
    main()
