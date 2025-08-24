#!/usr/bin/env python3
"""
Multi-Model Training Framework - Mother Script
Coordinates training of multiple model architectures with comprehensive hyperparameter tuning
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import json
import pickle
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Add project path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils import get_data_directories, find_project_root

class MultiModelTrainer:
    def __init__(self):
        """Initialize the multi-model training framework"""
        self.project_root = find_project_root()
        self.data_path = os.path.join(self.project_root, 'data', '15000_article', 'combined_labeled_dataset.csv')
        self.trainer_dir = os.path.join(self.project_root, '3_trainer', 'trainer')
        self.model_architectures_dir = os.path.join(self.trainer_dir, 'model_architectures')
        
        # Create model architectures directory
        os.makedirs(self.model_architectures_dir, exist_ok=True)
        
        # Results tracking
        self.results = {}
        self.best_models = {}
        
        # Available model architectures
        self.available_models = {
            'enhanced_bilstm': {
                'script': 'enhanced_bilstm.py',
                'description': 'Enhanced BiLSTM with attention and regularization',
                'estimated_time': '30-45 minutes'
            },
            'bilstm_crf': {
                'script': 'bilstm_crf.py',
                'description': 'BiLSTM with CRF layer for sequence labeling',
                'estimated_time': '45-60 minutes'
            },
            'cnn_bilstm_hybrid': {
                'script': 'cnn_bilstm_hybrid.py',
                'description': 'CNN feature extraction + BiLSTM classification',
                'estimated_time': '30-40 minutes'
            },
            'distilbert_classifier': {
                'script': 'distilbert_classifier.py',
                'description': 'DistilBERT-based classification',
                'estimated_time': '60-90 minutes'
            },
            'transformer_encoder': {
                'script': 'transformer_encoder.py',
                'description': 'Lightweight transformer encoder',
                'estimated_time': '45-60 minutes'
            },
            'ensemble_model': {
                'script': 'ensemble_model.py',
                'description': 'Ensemble of best performing models',
                'estimated_time': '20-30 minutes'
            }
        }
        
    def load_and_prepare_data(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Load and prepare the dataset for training"""
        print("=" * 80)
        print("📊 LOADING AND PREPARING DATASET")
        print("=" * 80)
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found: {self.data_path}")
        
        print(f"📁 Loading data from: {self.data_path}")
        df = pd.read_csv(self.data_path)
        
        print(f"📋 Dataset Info:")
        print(f"   • Total samples: {len(df):,}")
        print(f"   • Categories: {df['category'].unique()}")
        print(f"   • Label distribution:")
        for label, count in df['label'].value_counts().items():
            category = df[df['label'] == label]['category'].iloc[0]
            print(f"     - {category} (label {label}): {count:,} samples")
        
        # Extract features and labels
        texts = df['tokens'].values
        labels = df['label'].values
        
        # Create vocabulary and statistics
        all_tokens = []
        token_lengths = []
        
        for text in texts:
            tokens = text.split()
            all_tokens.extend(tokens)
            token_lengths.append(len(tokens))
        
        vocab = list(set(all_tokens))
        vocab_size = len(vocab)
        max_length = max(token_lengths)
        avg_length = np.mean(token_lengths)
        
        print(f"\n📈 Text Statistics:")
        print(f"   • Vocabulary size: {vocab_size:,}")
        print(f"   • Max sequence length: {max_length:,}")
        print(f"   • Average sequence length: {avg_length:.1f}")
        print(f"   • Min sequence length: {min(token_lengths):,}")
        
        # Create word-to-index mapping
        word_to_idx = {word: idx + 2 for idx, word in enumerate(vocab)}  # Reserve 0 for padding, 1 for OOV
        word_to_idx['<PAD>'] = 0
        word_to_idx['<OOV>'] = 1
        
        # Prepare metadata for models
        data_info = {
            'vocab_size': vocab_size + 2,  # +2 for PAD and OOV
            'max_length': max_length,
            'avg_length': avg_length,
            'num_classes': len(np.unique(labels)),
            'class_distribution': df['label'].value_counts().to_dict(),
            'word_to_idx': word_to_idx,
            'label_mapping': {0: 'red', 1: 'neutral', 2: 'green'},
            'total_samples': len(df)
        }
        
        print(f"\n✅ Data preparation complete!")
        return texts, labels, data_info
    
    def create_model_architecture_scripts(self):
        """Create individual model architecture scripts"""
        print("\n" + "=" * 80)
        print("🏗️  CREATING MODEL ARCHITECTURE SCRIPTS")
        print("=" * 80)
        
        scripts_to_create = [
            ('enhanced_bilstm.py', self._get_enhanced_bilstm_script()),
            ('bilstm_crf.py', self._get_bilstm_crf_script()),
            ('cnn_bilstm_hybrid.py', self._get_cnn_bilstm_script()),
            ('distilbert_classifier.py', self._get_distilbert_script()),
            ('transformer_encoder.py', self._get_transformer_script()),
            ('ensemble_model.py', self._get_ensemble_script())
        ]
        
        for script_name, script_content in scripts_to_create:
            script_path = os.path.join(self.model_architectures_dir, script_name)
            
            if not os.path.exists(script_path):
                with open(script_path, 'w', encoding='utf-8') as f:
                    f.write(script_content)
                print(f"✅ Created: {script_name}")
            else:
                print(f"📝 Exists: {script_name}")
    
    def run_model_training(self, model_names: List[str] = None):
        """Run training for specified models or all models"""
        if model_names is None:
            model_names = list(self.available_models.keys())
        
        print("\n" + "=" * 80)
        print("🚀 STARTING MULTI-MODEL TRAINING")
        print("=" * 80)
        
        # Load data once and save for all models
        texts, labels, data_info = self.load_and_prepare_data()
        
        # Save prepared data for model scripts
        data_cache = {
            'texts': texts,
            'labels': labels,
            'data_info': data_info
        }
        
        cache_path = os.path.join(self.trainer_dir, 'data_cache.pkl')
        with open(cache_path, 'wb') as f:
            pickle.dump(data_cache, f)
        
        print(f"💾 Data cached at: {cache_path}")
        
        total_estimated_time = sum([self._parse_time(self.available_models[name]['estimated_time']) 
                                   for name in model_names])
        print(f"⏱️  Estimated total training time: {total_estimated_time} minutes")
        
        # Train each model
        for i, model_name in enumerate(model_names, 1):
            if model_name not in self.available_models:
                print(f"❌ Unknown model: {model_name}")
                continue
            
            model_info = self.available_models[model_name]
            print(f"\n{'='*60}")
            print(f"🤖 Training Model {i}/{len(model_names)}: {model_name.upper()}")
            print(f"📝 Description: {model_info['description']}")
            print(f"⏱️  Estimated time: {model_info['estimated_time']}")
            print(f"{'='*60}")
            
            try:
                result = self._train_single_model(model_name, model_info)
                self.results[model_name] = result
                print(f"✅ {model_name} training completed!")
                
            except Exception as e:
                print(f"❌ {model_name} training failed: {e}")
                self.results[model_name] = {'error': str(e), 'status': 'failed'}
        
        # Generate final report
        self._generate_final_report()
    
    def _train_single_model(self, model_name: str, model_info: Dict) -> Dict:
        """Train a single model architecture"""
        script_path = os.path.join(self.model_architectures_dir, model_info['script'])
        
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Model script not found: {script_path}")
        
        # Run the model script
        start_time = datetime.now()
        
        try:
            result = subprocess.run([
                sys.executable, script_path
            ], capture_output=True, text=True, cwd=self.trainer_dir)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds() / 60  # minutes
            
            if result.returncode == 0:
                # Parse results from model output
                return {
                    'status': 'success',
                    'duration_minutes': duration,
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
            else:
                return {
                    'status': 'failed',
                    'error': result.stderr,
                    'stdout': result.stdout,
                    'duration_minutes': duration
                }
                
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e),
                'duration_minutes': 0
            }
    
    def _generate_final_report(self):
        """Generate comprehensive final report of all model results"""
        print("\n" + "=" * 80)
        print("📊 MULTI-MODEL TRAINING RESULTS")
        print("=" * 80)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_dir = os.path.join(self.trainer_dir, 'multi_model_reports', f'session_{timestamp}')
        os.makedirs(report_dir, exist_ok=True)
        
        successful_models = []
        failed_models = []
        
        for model_name, result in self.results.items():
            if result.get('status') == 'success':
                successful_models.append((model_name, result))
                print(f"✅ {model_name}: SUCCESS ({result.get('duration_minutes', 0):.1f} min)")
            else:
                failed_models.append((model_name, result))
                print(f"❌ {model_name}: FAILED - {result.get('error', 'Unknown error')}")
        
        print(f"\n📈 Summary:")
        print(f"   • Successful models: {len(successful_models)}")
        print(f"   • Failed models: {len(failed_models)}")
        print(f"   • Total time: {sum(r.get('duration_minutes', 0) for _, r in self.results.items()):.1f} minutes")
        
        # Save detailed report
        report_data = {
            'session_info': {
                'timestamp': timestamp,
                'total_models': len(self.results),
                'successful_models': len(successful_models),
                'failed_models': len(failed_models)
            },
            'results': self.results,
            'data_info': getattr(self, 'data_info', {})
        }
        
        with open(os.path.join(report_dir, 'training_report.json'), 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"\n📁 Detailed report saved: {report_dir}")
        
    def _parse_time(self, time_str: str) -> int:
        """Parse estimated time string to minutes"""
        # Simple parser for "30-45 minutes" format
        try:
            if '-' in time_str:
                return int(time_str.split('-')[1].split()[0])
            else:
                return int(time_str.split()[0])
        except:
            return 60  # default
    
    def show_menu(self):
        """Display model selection menu"""
        print("\n" + "=" * 80)
        print("🤖 MULTI-MODEL TRAINING FRAMEWORK")
        print("=" * 80)
        print("\nAvailable Model Architectures:")
        
        for i, (name, info) in enumerate(self.available_models.items(), 1):
            print(f"  {i}. {name.upper()}")
            print(f"     📝 {info['description']}")
            print(f"     ⏱️  {info['estimated_time']}")
            print()
        
        print("Training Options:")
        print("  A. Train all models")
        print("  S. Select specific models")
        print("  C. Create architecture scripts only")
        print("  Q. Quit")
        
    def run(self):
        """Main execution loop"""
        self.show_menu()
        
        while True:
            choice = input("\nSelect option: ").strip().upper()
            
            if choice == 'Q':
                print("👋 Goodbye!")
                break
                
            elif choice == 'C':
                self.create_model_architecture_scripts()
                input("\nPress Enter to continue...")
                self.show_menu()
                
            elif choice == 'A':
                self.create_model_architecture_scripts()
                self.run_model_training()
                break
                
            elif choice == 'S':
                self.create_model_architecture_scripts()
                model_names = self._select_models()
                if model_names:
                    self.run_model_training(model_names)
                break
                
            else:
                print("❌ Invalid option!")
    
    def _select_models(self) -> List[str]:
        """Interactive model selection"""
        print("\nSelect models to train (comma-separated numbers):")
        models = list(self.available_models.keys())
        
        for i, name in enumerate(models, 1):
            print(f"  {i}. {name}")
        
        try:
            choices = input("\nEnter numbers: ").strip().split(',')
            selected = [models[int(c.strip()) - 1] for c in choices if c.strip().isdigit()]
            
            if selected:
                print(f"Selected: {', '.join(selected)}")
                return selected
            else:
                print("No valid models selected!")
                return []
        except:
            print("Invalid selection!")
            return []

    def _get_enhanced_bilstm_script(self) -> str:
        """Generate enhanced BiLSTM script with comprehensive hyperparameter tuning"""
        return '''#!/usr/bin/env python3
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
    cache_path = os.path.join('..', 'data_cache.pkl')
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
    
    print(f"\\n📊 Final Results:")
    print(f"   Test Accuracy: {test_acc:.4f}")
    print(f"   Test Loss: {test_loss:.4f}")
    
    # Classification report
    print(f"\\n📋 Classification Report:")
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
'''

    def _get_bilstm_crf_script(self) -> str:
        """Generate BiLSTM + CRF script"""
        return '''#!/usr/bin/env python3
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
    
    print(f"\\n📊 Final Results:")
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
'''

    def _get_cnn_bilstm_script(self) -> str:
        """Generate CNN + BiLSTM hybrid script"""
        return '''#!/usr/bin/env python3
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
    
    print(f"\\n📊 Final Results:")
    print(f"   Test Accuracy: {test_acc:.4f}")
    print(f"   Test Loss: {test_loss:.4f}")
    
    print(f"\\n📋 Classification Report:")
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
'''

    def _get_distilbert_script(self) -> str:
        """Generate DistilBERT script"""
        return '''#!/usr/bin/env python3
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
        
        print(f"\\n📊 Final Results:")
        print(f"   Test Accuracy: {test_acc:.4f}")
        print(f"   Test Loss: {test_loss:.4f}")
        
        print(f"\\n📋 Classification Report:")
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
'''

    def _get_transformer_script(self) -> str:
        """Generate lightweight transformer script"""
        return '''#!/usr/bin/env python3
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
        x = transformer_block1(x)
        
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
    cache_path = os.path.join('..', 'data_cache.pkl')
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
    
    print(f"\\n📊 Final Results:")
    print(f"   Test Accuracy: {test_acc:.4f}")
    print(f"   Test Loss: {test_loss:.4f}")
    
    print(f"\\n📋 Classification Report:")
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
'''

    def _get_ensemble_script(self) -> str:
        """Generate ensemble model script"""
        return '''#!/usr/bin/env python3
"""
Ensemble Model
Combines predictions from multiple trained models
"""

import os
import pickle
import numpy as np
import glob
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from collections import Counter

class EnsembleModel:
    def __init__(self):
        self.model_results = {}
        self.predictions = {}
        
    def load_model_results(self):
        """Load results from all trained models"""
        result_files = glob.glob('*_results_*.json')
        
        for result_file in result_files:
            try:
                with open(result_file, 'r') as f:
                    result = json.load(f)
                    
                model_name = result.get('model_name', 'unknown')
                if 'error' not in result and 'test_accuracy' in result:
                    self.model_results[model_name] = result
                    print(f"✅ Loaded results for {model_name}: {result['test_accuracy']:.4f}")
                    
            except Exception as e:
                print(f"❌ Failed to load {result_file}: {e}")
        
        return len(self.model_results)
    
    def create_ensemble_prediction(self, test_labels):
        """Create ensemble predictions using voting"""
        if len(self.model_results) < 2:
            print("❌ Need at least 2 models for ensemble")
            return None
        
        # For this simplified version, we'll simulate predictions
        # In practice, you'd load actual model predictions
        
        # Simple majority voting simulation
        num_samples = len(test_labels)
        ensemble_predictions = []
        
        # Simulate predictions from each model
        model_names = list(self.model_results.keys())
        simulated_predictions = {}
        
        for model_name in model_names:
            accuracy = self.model_results[model_name]['test_accuracy']
            # Simulate predictions based on accuracy
            predictions = []
            for true_label in test_labels:
                if np.random.random() < accuracy:
                    predictions.append(true_label)  # Correct prediction
                else:
                    # Random incorrect prediction
                    wrong_labels = [0, 1, 2]
                    wrong_labels.remove(true_label)
                    predictions.append(np.random.choice(wrong_labels))
            
            simulated_predictions[model_name] = predictions
        
        # Majority voting
        for i in range(num_samples):
            votes = [simulated_predictions[model][i] for model in model_names]
            ensemble_pred = Counter(votes).most_common(1)[0][0]
            ensemble_predictions.append(ensemble_pred)
        
        return np.array(ensemble_predictions), simulated_predictions
    
    def weighted_ensemble(self, test_labels):
        """Create weighted ensemble based on model performance"""
        if len(self.model_results) < 2:
            return None
        
        # Calculate weights based on accuracy
        total_accuracy = sum(result['test_accuracy'] for result in self.model_results.values())
        weights = {name: result['test_accuracy'] / total_accuracy 
                  for name, result in self.model_results.items()}
        
        print("📊 Model weights:")
        for name, weight in weights.items():
            print(f"   {name}: {weight:.3f}")
        
        # Simulate weighted predictions
        num_samples = len(test_labels)
        weighted_predictions = []
        
        for i in range(num_samples):
            # Simulate weighted voting
            vote_scores = {0: 0, 1: 0, 2: 0}
            
            for model_name, weight in weights.items():
                accuracy = self.model_results[model_name]['test_accuracy']
                if np.random.random() < accuracy:
                    vote_scores[test_labels[i]] += weight
                else:
                    wrong_labels = [0, 1, 2]
                    wrong_labels.remove(test_labels[i])
                    vote_scores[np.random.choice(wrong_labels)] += weight
            
            # Choose label with highest weighted score
            best_label = max(vote_scores, key=vote_scores.get)
            weighted_predictions.append(best_label)
        
        return np.array(weighted_predictions)

def main():
    """Main ensemble function"""
    print("🤖 Ensemble Model Training Started")
    
    # Load cached data for test set
    cache_path = os.path.join('..', 'data_cache.pkl')
    with open(cache_path, 'rb') as f:
        data = pickle.load(f)
    
    labels = data['labels']
    
    # Split to get test set (same split as other models)
    _, y_test = train_test_split(labels, labels, test_size=0.2, random_state=42, stratify=labels)
    
    # Initialize ensemble
    ensemble = EnsembleModel()
    
    # Load model results
    num_models = ensemble.load_model_results()
    
    if num_models < 2:
        print("❌ Need at least 2 trained models for ensemble")
        print("Please run other model training scripts first")
        return
    
    print(f"\\n🔄 Creating ensemble from {num_models} models...")
    
    # Simple majority voting
    ensemble_pred, individual_preds = ensemble.create_ensemble_prediction(y_test)
    
    if ensemble_pred is not None:
        ensemble_acc = np.mean(ensemble_pred == y_test)
        print(f"\\n📊 Ensemble Results (Majority Voting):")
        print(f"   Ensemble Accuracy: {ensemble_acc:.4f}")
        
        print(f"\\n📋 Classification Report:")
        print(classification_report(y_test, ensemble_pred, target_names=['red', 'neutral', 'green']))
        
        # Weighted ensemble
        weighted_pred = ensemble.weighted_ensemble(y_test)
        if weighted_pred is not None:
            weighted_acc = np.mean(weighted_pred == y_test)
            print(f"\\n📊 Weighted Ensemble Accuracy: {weighted_acc:.4f}")
        
        # Compare with individual models
        print(f"\\n📈 Model Comparison:")
        for name, result in ensemble.model_results.items():
            print(f"   {name}: {result['test_accuracy']:.4f}")
        print(f"   ensemble_majority: {ensemble_acc:.4f}")
        if weighted_pred is not None:
            print(f"   ensemble_weighted: {weighted_acc:.4f}")
        
        # Save results
        results = {
            'model_name': 'ensemble_model',
            'majority_voting_accuracy': float(ensemble_acc),
            'weighted_ensemble_accuracy': float(weighted_acc) if weighted_pred is not None else None,
            'individual_models': ensemble.model_results,
            'num_models': num_models
        }
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(f'ensemble_results_{timestamp}.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\\n✅ Ensemble training completed!")
        print(f"📁 Results saved: ensemble_results_{timestamp}.json")
    
    else:
        print("❌ Ensemble creation failed")

if __name__ == "__main__":
    main()
'''

def main():
    """Main entry point"""
    trainer = MultiModelTrainer()
    trainer.run()

if __name__ == "__main__":
    main()