import os
import pandas as pd
import numpy as np
import pickle
import json
import shutil
from datetime import datetime
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Embedding, LSTM, Bidirectional, Dense, Dropout, Input,
    GlobalMaxPooling1D, Concatenate, TimeDistributed
)
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# Check if transformers is available and compatible
try:
    from transformers import DistilBertTokenizer, TFDistilBertModel
    TRANSFORMERS_AVAILABLE = True
except (ImportError, ValueError) as e:
    print(f"Warning: transformers not available or incompatible - {e}")
    TRANSFORMERS_AVAILABLE = False
import matplotlib.pyplot as plt
import seaborn as sns

# Check if tensorflow-addons is available for CRF
try:
    import tensorflow_addons as tfa
    CRF_AVAILABLE = True
except ImportError:
    print("Warning: tensorflow-addons not available. BiLSTM-CRF model will be skipped.")
    CRF_AVAILABLE = False

class HyperparameterTuningTrainer:
    def __init__(self, dataset_path, model_output_dir):
        """
        Initialize Hyperparameter Tuning trainer for multiple architectures
        
        Args:
            dataset_path: Path to the labeled dataset CSV
            model_output_dir: Directory to save trained models and artifacts
        """
        self.dataset_path = dataset_path
        self.model_output_dir = model_output_dir
        self.word_to_index = None
        self.index_to_word = None
        self.max_length = None
        self.vocab_size = None
        
        # Create output directory
        os.makedirs(model_output_dir, exist_ok=True)
        
        # Label mapping
        self.label_mapping = {0: 'red', 1: 'neutral', 2: 'green'}
        
        # Experiment tracking
        self.all_experiments = []
        self.best_model_per_architecture = {}
        
        # Base configuration
        self.base_config = {
            'vocab_size': 20000,
            'max_length': 1500,
            'test_size': 0.2,
            'validation_split': 0.2,
            'epochs': 30,
            'batch_size': 16
        }
        
        print(f"📁 Model output directory: {model_output_dir}")
        
    def load_and_prepare_data(self):
        """Load and prepare the dataset"""
        print("Loading dataset...")
        
        df = pd.read_csv(self.dataset_path)
        label_dist = df['category'].value_counts()
        
        print(f"Dataset shape: {df.shape}")
        print(f"Label distribution:\n{label_dist}")
        
        texts = df['tokens'].values
        labels = df['label'].values
        
        return texts, labels
    
    def preprocess_texts(self, texts, labels, vocab_size=20000, max_length=1500):
        """Preprocess tokenized texts for training"""
        print("Preprocessing texts...")
        
        # Convert tokenized strings to list of tokens
        tokenized_texts = []
        for text in texts:
            if isinstance(text, str):
                tokens = text.strip().split()
                tokenized_texts.append(tokens)
            else:
                tokenized_texts.append([])
        
        # Build vocabulary
        word_freq = {}
        for tokens in tokenized_texts:
            for token in tokens:
                word_freq[token] = word_freq.get(token, 0) + 1
        
        # Sort by frequency and take top vocab_size-1
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        vocab_words = [word for word, freq in sorted_words[:vocab_size-1]]
        
        # Create word to index mapping
        self.word_to_index = {'<PAD>': 0, '<OOV>': 1}
        for i, word in enumerate(vocab_words, start=2):
            self.word_to_index[word] = i
        
        self.index_to_word = {idx: word for word, idx in self.word_to_index.items()}
        
        # Convert tokens to sequences
        sequences = []
        oov_count = 0
        total_tokens = 0
        
        for tokens in tokenized_texts:
            sequence = []
            for token in tokens:
                total_tokens += 1
                if token in self.word_to_index:
                    sequence.append(self.word_to_index[token])
                else:
                    sequence.append(self.word_to_index['<OOV>'])
                    oov_count += 1
            sequences.append(sequence)
        
        print(f"OOV rate: {oov_count/total_tokens*100:.2f}%")
        
        # Pad sequences
        padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
        
        # Convert labels to categorical
        categorical_labels = to_categorical(labels, num_classes=3)
        
        # Store parameters
        self.vocab_size = len(self.word_to_index)
        self.max_length = max_length
        
        print(f"Built vocabulary size: {self.vocab_size}")
        print(f"Max sequence length: {self.max_length}")
        
        return padded_sequences, categorical_labels
    
    def build_bilstm_model(self, embedding_dim, lstm_units, dropout_rate, dense_units=32):
        """Build standard BiLSTM model"""
        model = Sequential([
            Embedding(
                input_dim=self.vocab_size,
                output_dim=embedding_dim,
                input_length=self.max_length,
                mask_zero=True
            ),
            Bidirectional(LSTM(lstm_units, dropout=dropout_rate)),
            Dense(dense_units, activation='relu'),
            Dropout(dropout_rate),
            Dense(3, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_enhanced_bilstm_model(self, embedding_dim, lstm_units, dropout_rate, dense_units=64):
        """Build enhanced BiLSTM with multiple layers"""
        model = Sequential([
            Embedding(
                input_dim=self.vocab_size,
                output_dim=embedding_dim,
                input_length=self.max_length,
                mask_zero=True
            ),
            Bidirectional(LSTM(lstm_units, dropout=dropout_rate, return_sequences=True)),
            Bidirectional(LSTM(lstm_units//2, dropout=dropout_rate)),
            Dense(dense_units, activation='relu'),
            Dropout(dropout_rate),
            Dense(dense_units//2, activation='relu'),
            Dropout(dropout_rate/2),
            Dense(3, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_bilstm_cnn_hybrid(self, embedding_dim, lstm_units, dropout_rate, filters=64):
        """Build BiLSTM-CNN hybrid model"""
        from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D
        
        input_layer = Input(shape=(self.max_length,))
        
        # Embedding layer
        embedding = Embedding(
            input_dim=self.vocab_size,
            output_dim=embedding_dim,
            input_length=self.max_length,
            mask_zero=True
        )(input_layer)
        
        # BiLSTM branch
        bilstm = Bidirectional(LSTM(lstm_units, dropout=dropout_rate))(embedding)
        
        # CNN branch
        conv1 = Conv1D(filters=filters, kernel_size=3, activation='relu')(embedding)
        pool1 = GlobalMaxPooling1D()(conv1)
        
        conv2 = Conv1D(filters=filters, kernel_size=4, activation='relu')(embedding)
        pool2 = GlobalMaxPooling1D()(conv2)
        
        # Concatenate features
        concat = Concatenate()([bilstm, pool1, pool2])
        
        # Dense layers
        dense1 = Dense(128, activation='relu')(concat)
        dropout1 = Dropout(dropout_rate)(dense1)
        
        dense2 = Dense(64, activation='relu')(dropout1)
        dropout2 = Dropout(dropout_rate/2)(dense2)
        
        output = Dense(3, activation='softmax')(dropout2)
        
        model = Model(inputs=input_layer, outputs=output)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_bilstm_crf_model(self, embedding_dim, lstm_units, dropout_rate):
        """Build BiLSTM-CRF model (requires tensorflow-addons)"""
        if not CRF_AVAILABLE:
            print("Skipping BiLSTM-CRF: tensorflow-addons not available")
            return None
            
        input_layer = Input(shape=(self.max_length,))
        
        embedding = Embedding(
            input_dim=self.vocab_size,
            output_dim=embedding_dim,
            input_length=self.max_length,
            mask_zero=True
        )(input_layer)
        
        bilstm = Bidirectional(LSTM(lstm_units, dropout=dropout_rate, return_sequences=True))(embedding)
        
        # Dense layer before CRF
        dense = TimeDistributed(Dense(3, activation='relu'))(bilstm)
        
        # CRF layer
        crf = tfa.layers.CRF(3)  # 3 classes
        crf_output = crf(dense)
        
        model = Model(inputs=input_layer, outputs=crf_output)
        
        # Custom compile for CRF
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=crf.loss_function,
            metrics=[crf.accuracy]
        )
        
        return model
    
    def build_distilbert_model(self, dropout_rate=0.3, learning_rate=2e-5):
        """Build DistilBERT model for classification"""
        if not TRANSFORMERS_AVAILABLE:
            print("Skipping DistilBERT: transformers not available or incompatible")
            return None
            
        try:
            # Initialize DistilBERT
            distilbert = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
            
            input_ids = Input(shape=(self.max_length,), dtype=tf.int32, name='input_ids')
            attention_mask = Input(shape=(self.max_length,), dtype=tf.int32, name='attention_mask')
            
            # DistilBERT outputs
            distilbert_output = distilbert([input_ids, attention_mask])
            sequence_output = distilbert_output.last_hidden_state
            
            # Global pooling
            pooled_output = GlobalMaxPooling1D()(sequence_output)
            
            # Classification head
            dense1 = Dense(256, activation='relu')(pooled_output)
            dropout1 = Dropout(dropout_rate)(dense1)
            
            dense2 = Dense(128, activation='relu')(dropout1)
            dropout2 = Dropout(dropout_rate/2)(dense2)
            
            output = Dense(3, activation='softmax')(dropout2)
            
            model = Model(inputs=[input_ids, attention_mask], outputs=output)
            model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            return model
            
        except Exception as e:
            print(f"Warning: Failed to build DistilBERT model - {e}")
            return None
    
    def prepare_distilbert_data(self, texts, labels):
        """Prepare data for DistilBERT model"""
        if not TRANSFORMERS_AVAILABLE:
            return None
            
        try:
            tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            
            # Tokenize texts
            input_ids = []
            attention_masks = []
            
            for text in texts:
                if isinstance(text, str):
                    # Join tokens back to string for DistilBERT
                    text_str = text.replace(' ', ' ')  # Ensure proper spacing
                else:
                    text_str = ""
                
                encoding = tokenizer(
                    text_str,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='tf'
                )
                
                input_ids.append(encoding['input_ids'][0])
                attention_masks.append(encoding['attention_mask'][0])
            
            return {
                'input_ids': tf.stack(input_ids),
                'attention_mask': tf.stack(attention_masks),
                'labels': to_categorical(labels, num_classes=3)
            }
            
        except Exception as e:
            print(f"Warning: Failed to prepare DistilBERT data - {e}")
            return None
    
    def get_architecture_configs(self):
        """Define architecture configurations and hyperparameters"""
        configs = {
            'bilstm': {
                'name': 'BiLSTM',
                'builder_func': self.build_bilstm_model,
                'hyperparams': [
                    # Config 1: Baseline
                    {'embedding_dim': 128, 'lstm_units': 64, 'dropout_rate': 0.3, 'dense_units': 32},
                    # Config 2: Larger model
                    {'embedding_dim': 256, 'lstm_units': 128, 'dropout_rate': 0.4, 'dense_units': 64},
                    # Config 3: Compact model
                    {'embedding_dim': 100, 'lstm_units': 32, 'dropout_rate': 0.2, 'dense_units': 16}
                ]
            },
            'enhanced_bilstm': {
                'name': 'Enhanced BiLSTM',
                'builder_func': self.build_enhanced_bilstm_model,
                'hyperparams': [
                    # Config 1: Moderate complexity
                    {'embedding_dim': 128, 'lstm_units': 64, 'dropout_rate': 0.3, 'dense_units': 64},
                    # Config 2: High complexity
                    {'embedding_dim': 256, 'lstm_units': 128, 'dropout_rate': 0.4, 'dense_units': 128},
                    # Config 3: Conservative
                    {'embedding_dim': 100, 'lstm_units': 48, 'dropout_rate': 0.25, 'dense_units': 48}
                ]
            },
            'bilstm_cnn_hybrid': {
                'name': 'BiLSTM-CNN Hybrid',
                'builder_func': self.build_bilstm_cnn_hybrid,
                'hyperparams': [
                    # Config 1: Balanced
                    {'embedding_dim': 128, 'lstm_units': 64, 'dropout_rate': 0.3, 'filters': 64},
                    # Config 2: More filters
                    {'embedding_dim': 150, 'lstm_units': 80, 'dropout_rate': 0.35, 'filters': 96},
                    # Config 3: Lightweight
                    {'embedding_dim': 100, 'lstm_units': 48, 'dropout_rate': 0.25, 'filters': 48}
                ]
            }
        }
        
        # Add BiLSTM-CRF if available
        if CRF_AVAILABLE:
            configs['bilstm_crf'] = {
                'name': 'BiLSTM-CRF',
                'builder_func': self.build_bilstm_crf_model,
                'hyperparams': [
                    # Config 1: Standard
                    {'embedding_dim': 128, 'lstm_units': 64, 'dropout_rate': 0.3},
                    # Config 2: Larger
                    {'embedding_dim': 200, 'lstm_units': 96, 'dropout_rate': 0.35},
                    # Config 3: Compact
                    {'embedding_dim': 100, 'lstm_units': 48, 'dropout_rate': 0.25}
                ]
            }
        
        # Add DistilBERT if transformers is available
        if TRANSFORMERS_AVAILABLE:
            configs['distilbert'] = {
                'name': 'DistilBERT',
                'builder_func': self.build_distilbert_model,
                'hyperparams': [
                    # Config 1: Standard
                    {'dropout_rate': 0.3, 'learning_rate': 2e-5},
                    # Config 2: Higher dropout
                    {'dropout_rate': 0.4, 'learning_rate': 1e-5},
                    # Config 3: Lower dropout, higher LR
                    {'dropout_rate': 0.2, 'learning_rate': 3e-5}
                ]
            }
        
        return configs
    
    def train_single_model(self, architecture_name, config_idx, model_builder, hyperparams, X, y, distilbert_data=None):
        """Train a single model configuration"""
        print(f"\n🔧 Training {architecture_name} - Config {config_idx + 1}")
        print(f"   Hyperparams: {hyperparams}")
        
        experiment_id = f"{architecture_name}_config_{config_idx + 1}"
        start_time = datetime.now()
        
        try:
            # Build model
            if architecture_name == 'distilbert' and distilbert_data is not None:
                model = model_builder(**hyperparams)
                if model is None:
                    return None
                
                # Use DistilBERT data
                X_train, X_test, y_train, y_test = train_test_split(
                    distilbert_data['input_ids'], distilbert_data['labels'],
                    test_size=self.base_config['test_size'], random_state=42,
                    stratify=distilbert_data['labels'].argmax(axis=1)
                )
                
                attention_train, attention_test = train_test_split(
                    distilbert_data['attention_mask'],
                    test_size=self.base_config['test_size'], random_state=42,
                    stratify=distilbert_data['labels'].argmax(axis=1)
                )
                
                train_data = ([X_train, attention_train], y_train)
                test_data = ([X_test, attention_test], y_test)
                
            else:
                model = model_builder(**hyperparams)
                if model is None:
                    return None
                
                # Standard data split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=self.base_config['test_size'], random_state=42,
                    stratify=y.argmax(axis=1)
                )
                
                train_data = (X_train, y_train)
                test_data = (X_test, y_test)
            
            # Compute class weights
            y_train_labels = train_data[1].argmax(axis=1)
            class_weights = compute_class_weight(
                'balanced', classes=np.unique(y_train_labels), y=y_train_labels
            )
            class_weight_dict = {i: class_weights[i] for i in range(3)}
            
            # Callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_accuracy',
                    patience=10,
                    restore_best_weights=True,
                    min_delta=0.01
                ),
                ReduceLROnPlateau(
                    monitor='val_accuracy',
                    factor=0.2,
                    patience=5,
                    min_lr=0.00001,
                    verbose=0
                )
            ]
            
            # Train model
            history = model.fit(
                train_data[0], train_data[1],
                validation_split=self.base_config['validation_split'],
                epochs=self.base_config['epochs'],
                batch_size=self.base_config['batch_size'],
                class_weight=class_weight_dict,
                callbacks=callbacks,
                verbose=0
            )
            
            # Evaluate
            test_loss, test_accuracy = model.evaluate(test_data[0], test_data[1], verbose=0)
            
            # Predictions for detailed metrics
            y_pred = model.predict(test_data[0], verbose=0)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true_classes = np.argmax(test_data[1], axis=1)
            
            # Classification report
            target_names = [self.label_mapping[i] for i in range(3)]
            class_report_dict = classification_report(
                y_true_classes, y_pred_classes, target_names=target_names,
                output_dict=True, zero_division=0
            )
            
            end_time = datetime.now()
            training_time = (end_time - start_time).total_seconds()
            
            # Store experiment results
            experiment = {
                'experiment_id': experiment_id,
                'architecture': architecture_name,
                'config_index': config_idx + 1,
                'hyperparams': hyperparams,
                'test_accuracy': float(test_accuracy),
                'test_loss': float(test_loss),
                'training_time_seconds': training_time,
                'best_val_accuracy': float(max(history.history['val_accuracy'])),
                'epochs_trained': len(history.history['accuracy']),
                'classification_report': class_report_dict,
                'confusion_matrix': confusion_matrix(y_true_classes, y_pred_classes).tolist(),
                'training_history': {
                    'accuracy': [float(x) for x in history.history['accuracy']],
                    'val_accuracy': [float(x) for x in history.history['val_accuracy']],
                    'loss': [float(x) for x in history.history['loss']],
                    'val_loss': [float(x) for x in history.history['val_loss']]
                },
                'timestamp': start_time.isoformat()
            }
            
            print(f"   ✅ Test Accuracy: {test_accuracy:.4f} | Val Accuracy: {max(history.history['val_accuracy']):.4f} | Time: {training_time:.1f}s")
            
            # Update best model for this architecture
            if (architecture_name not in self.best_model_per_architecture or 
                test_accuracy > self.best_model_per_architecture[architecture_name]['test_accuracy']):
                
                self.best_model_per_architecture[architecture_name] = experiment.copy()
                
                # Save best model
                model_filename = f"{experiment_id}_best.h5"
                model_path = os.path.join(self.model_output_dir, model_filename)
                model.save(model_path)
                experiment['model_path'] = model_path
                
                print(f"   🌟 New best model for {architecture_name}!")
            
            return experiment
            
        except Exception as e:
            print(f"   ❌ Failed: {str(e)}")
            return {
                'experiment_id': experiment_id,
                'architecture': architecture_name,
                'config_index': config_idx + 1,
                'hyperparams': hyperparams,
                'error': str(e),
                'timestamp': start_time.isoformat()
            }
    
    def run_hyperparameter_tuning(self):
        """Run hyperparameter tuning for all architectures"""
        print("=== Starting Hyperparameter Tuning ===")
        
        # Load and prepare data
        texts, labels = self.load_and_prepare_data()
        X, y = self.preprocess_texts(
            texts, labels,
            self.base_config['vocab_size'],
            self.base_config['max_length']
        )
        
        # Prepare DistilBERT data if needed
        distilbert_data = None
        if TRANSFORMERS_AVAILABLE:
            distilbert_data = self.prepare_distilbert_data(texts, labels)
        
        # Get all configurations
        configs = self.get_architecture_configs()
        
        print(f"\n📊 Starting experiments for {len(configs)} architectures:")
        for arch_name, arch_config in configs.items():
            print(f"   - {arch_config['name']}: {len(arch_config['hyperparams'])} configurations")
        
        total_experiments = sum(len(config['hyperparams']) for config in configs.values())
        print(f"   Total experiments: {total_experiments}")
        
        # Run all experiments
        experiment_count = 0
        for arch_name, arch_config in configs.items():
            print(f"\n🏗️ Architecture: {arch_config['name']}")
            
            for config_idx, hyperparams in enumerate(arch_config['hyperparams']):
                experiment_count += 1
                print(f"\nExperiment {experiment_count}/{total_experiments}")
                
                experiment = self.train_single_model(
                    arch_name, config_idx, arch_config['builder_func'],
                    hyperparams, X, y, distilbert_data
                )
                
                if experiment:
                    self.all_experiments.append(experiment)
        
        # Generate comprehensive report
        self.generate_tuning_report()
        
        print("\n=== Hyperparameter Tuning Complete! ===")
        self.print_summary()
    
    def generate_tuning_report(self):
        """Generate comprehensive hyperparameter tuning report"""
        print("\nGenerating hyperparameter tuning report...")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_dir = os.path.join(self.model_output_dir, f'hyperparameter_tuning_{timestamp}')
        os.makedirs(report_dir, exist_ok=True)
        
        # Save detailed results
        results_path = os.path.join(report_dir, 'all_experiments.json')
        with open(results_path, 'w') as f:
            json.dump({
                'experiments': self.all_experiments,
                'best_per_architecture': self.best_model_per_architecture,
                'base_config': self.base_config,
                'timestamp': timestamp
            }, f, indent=2)
        
        # Generate markdown report
        report_md = self._generate_tuning_markdown()
        report_path = os.path.join(report_dir, 'tuning_report.md')
        with open(report_path, 'w') as f:
            f.write(report_md)
        
        # Generate comparison plots
        self._generate_comparison_plots(report_dir)
        
        print(f"📋 Hyperparameter tuning report saved to: {report_dir}")
        return report_dir
    
    def _generate_tuning_markdown(self):
        """Generate markdown report for hyperparameter tuning"""
        successful_experiments = [exp for exp in self.all_experiments if 'error' not in exp]
        failed_experiments = [exp for exp in self.all_experiments if 'error' in exp]
        
        # Find overall best model
        overall_best = max(successful_experiments, key=lambda x: x['test_accuracy']) if successful_experiments else None
        
        md_content = f"""# Hyperparameter Tuning Report

Generated on: {datetime.now().isoformat()}

---

## 📊 Experiment Summary

| Metric | Value |
|--------|-------|
| **Total Experiments** | {len(self.all_experiments)} |
| **Successful Experiments** | {len(successful_experiments)} |
| **Failed Experiments** | {len(failed_experiments)} |
| **Architectures Tested** | {len(self.best_model_per_architecture)} |

### Base Configuration
| Parameter | Value |
|-----------|-------|
| **Vocabulary Size** | {self.base_config['vocab_size']:,} |
| **Max Sequence Length** | {self.base_config['max_length']} |
| **Training Epochs** | {self.base_config['epochs']} |
| **Batch Size** | {self.base_config['batch_size']} |
| **Test Split** | {self.base_config['test_size']} |
| **Validation Split** | {self.base_config['validation_split']} |

---

## 🏆 Overall Best Model
"""
        
        if overall_best:
            md_content += f"""
| Metric | Value |
|--------|-------|
| **Architecture** | {overall_best['architecture']} |
| **Configuration** | Config {overall_best['config_index']} |
| **Test Accuracy** | {overall_best['test_accuracy']:.4f} |
| **Validation Accuracy** | {overall_best['best_val_accuracy']:.4f} |
| **Training Time** | {overall_best['training_time_seconds']:.1f} seconds |
| **Epochs Trained** | {overall_best['epochs_trained']} |

### Best Model Hyperparameters
```json
{json.dumps(overall_best['hyperparams'], indent=2)}
```
"""
        else:
            md_content += "\n❌ No successful experiments found.\n"
        
        # Best per architecture
        md_content += "\n---\n\n## 🎯 Best Model Per Architecture\n\n"
        
        for arch_name, best_model in self.best_model_per_architecture.items():
            md_content += f"""### {best_model['architecture']}
| Metric | Value |
|--------|-------|
| **Test Accuracy** | {best_model['test_accuracy']:.4f} |
| **Validation Accuracy** | {best_model['best_val_accuracy']:.4f} |
| **Configuration** | Config {best_model['config_index']} |
| **Training Time** | {best_model['training_time_seconds']:.1f}s |

**Hyperparameters:**
```json
{json.dumps(best_model['hyperparams'], indent=2)}
```

"""
        
        # Detailed results table
        md_content += "\n---\n\n## 📈 All Experiment Results\n\n"
        md_content += "| Architecture | Config | Test Acc | Val Acc | Time (s) | Status |\n"
        md_content += "|--------------|--------|----------|---------|----------|--------|\n"
        
        for exp in self.all_experiments:
            if 'error' in exp:
                md_content += f"| {exp['architecture']} | {exp['config_index']} | - | - | - | ❌ Failed |\n"
            else:
                md_content += f"| {exp['architecture']} | {exp['config_index']} | {exp['test_accuracy']:.4f} | {exp['best_val_accuracy']:.4f} | {exp['training_time_seconds']:.1f} | ✅ Success |\n"
        
        # Performance analysis
        if successful_experiments:
            md_content += "\n---\n\n## 📊 Performance Analysis\n\n"
            
            accuracies = [exp['test_accuracy'] for exp in successful_experiments]
            times = [exp['training_time_seconds'] for exp in successful_experiments]
            
            md_content += f"""### Statistical Summary
| Metric | Value |
|--------|-------|
| **Mean Test Accuracy** | {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f} |
| **Best Test Accuracy** | {np.max(accuracies):.4f} |
| **Worst Test Accuracy** | {np.min(accuracies):.4f} |
| **Mean Training Time** | {np.mean(times):.1f} ± {np.std(times):.1f} seconds |

### Architecture Performance Ranking
"""
            
            # Rank architectures by best performance
            arch_performance = []
            for arch_name, best_model in self.best_model_per_architecture.items():
                arch_performance.append((arch_name, best_model['test_accuracy'], best_model['training_time_seconds']))
            
            arch_performance.sort(key=lambda x: x[1], reverse=True)
            
            for i, (arch_name, accuracy, time) in enumerate(arch_performance, 1):
                emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
                md_content += f"{emoji} **{arch_name}**: {accuracy:.4f} accuracy ({time:.1f}s)\n"
        
        # Failed experiments
        if failed_experiments:
            md_content += "\n---\n\n## ⚠️ Failed Experiments\n\n"
            for exp in failed_experiments:
                md_content += f"- **{exp['architecture']} Config {exp['config_index']}**: {exp['error']}\n"
        
        # Recommendations
        md_content += f"""

---

## 💡 Recommendations

### For Production Use:
1. **Deploy the overall best model**: {overall_best['architecture'] if overall_best else 'N/A'} with {overall_best['test_accuracy']:.4f if overall_best else 'N/A'} accuracy
2. **Consider ensemble methods** combining the top 3 performing models
3. **Monitor performance** on new data and retrain if accuracy drops

### For Further Optimization:
1. **Fine-tune the best architecture** with more granular hyperparameter search
2. **Try different optimizers** (AdamW, RMSprop) for the top models
3. **Experiment with learning rate scheduling** and warmup strategies
4. **Consider data augmentation** techniques to improve robustness

### For Computational Efficiency:
- **Fast inference**: Use the most compact model with acceptable accuracy
- **Batch processing**: Optimize batch sizes for your deployment environment
- **Model quantization**: Consider INT8 quantization for mobile/edge deployment

---

*Report generated by Hyperparameter Tuning Trainer*
"""
        
        return md_content
    
    def _generate_comparison_plots(self, report_dir):
        """Generate comparison plots for different models"""
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        
        successful_experiments = [exp for exp in self.all_experiments if 'error' not in exp]
        
        if not successful_experiments:
            return
        
        # Prepare data for plotting
        architectures = [exp['architecture'] for exp in successful_experiments]
        test_accuracies = [exp['test_accuracy'] for exp in successful_experiments]
        val_accuracies = [exp['best_val_accuracy'] for exp in successful_experiments]
        training_times = [exp['training_time_seconds'] for exp in successful_experiments]
        
        # 1. Accuracy comparison
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        arch_names = list(set(architectures))
        arch_colors = plt.cm.tab10(np.linspace(0, 1, len(arch_names)))
        
        for i, arch in enumerate(arch_names):
            arch_indices = [j for j, a in enumerate(architectures) if a == arch]
            arch_test_acc = [test_accuracies[j] for j in arch_indices]
            arch_val_acc = [val_accuracies[j] for j in arch_indices]
            
            plt.scatter([i] * len(arch_test_acc), arch_test_acc, 
                       c=[arch_colors[i]], s=100, alpha=0.7, label=f'{arch} (test)')
            plt.scatter([i+0.1] * len(arch_val_acc), arch_val_acc, 
                       c=[arch_colors[i]], s=50, alpha=0.5, marker='x')
        
        plt.xlabel('Architecture')
        plt.ylabel('Accuracy')
        plt.title('Test Accuracy by Architecture')
        plt.xticks(range(len(arch_names)), arch_names, rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 2. Training time comparison
        plt.subplot(2, 2, 2)
        for i, arch in enumerate(arch_names):
            arch_indices = [j for j, a in enumerate(architectures) if a == arch]
            arch_times = [training_times[j] for j in arch_indices]
            plt.scatter([i] * len(arch_times), arch_times, 
                       c=[arch_colors[i]], s=100, alpha=0.7)
        
        plt.xlabel('Architecture')
        plt.ylabel('Training Time (seconds)')
        plt.title('Training Time by Architecture')
        plt.xticks(range(len(arch_names)), arch_names, rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 3. Accuracy vs Time scatter
        plt.subplot(2, 2, 3)
        for i, arch in enumerate(arch_names):
            arch_indices = [j for j, a in enumerate(architectures) if a == arch]
            arch_test_acc = [test_accuracies[j] for j in arch_indices]
            arch_times = [training_times[j] for j in arch_indices]
            plt.scatter(arch_times, arch_test_acc, 
                       c=[arch_colors[i]], s=100, alpha=0.7, label=arch)
        
        plt.xlabel('Training Time (seconds)')
        plt.ylabel('Test Accuracy')
        plt.title('Accuracy vs Training Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Best model per architecture
        plt.subplot(2, 2, 4)
        best_archs = []
        best_accs = []
        for arch_name, best_model in self.best_model_per_architecture.items():
            best_archs.append(arch_name)
            best_accs.append(best_model['test_accuracy'])
        
        bars = plt.bar(best_archs, best_accs, color=arch_colors[:len(best_archs)])
        plt.xlabel('Architecture')
        plt.ylabel('Best Test Accuracy')
        plt.title('Best Model Per Architecture')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, acc in zip(bars, best_accs):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plot_path = os.path.join(report_dir, 'model_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Comparison plots saved to: {plot_path}")
    
    def print_summary(self):
        """Print a summary of the hyperparameter tuning results"""
        successful_experiments = [exp for exp in self.all_experiments if 'error' not in exp]
        
        if not successful_experiments:
            print("❌ No successful experiments found!")
            return
        
        overall_best = max(successful_experiments, key=lambda x: x['test_accuracy'])
        
        print(f"\n🎉 HYPERPARAMETER TUNING SUMMARY")
        print(f"{'='*50}")
        print(f"📊 Total Experiments: {len(self.all_experiments)}")
        print(f"✅ Successful: {len(successful_experiments)}")
        print(f"❌ Failed: {len(self.all_experiments) - len(successful_experiments)}")
        print(f"\n🏆 OVERALL BEST MODEL:")
        print(f"   Architecture: {overall_best['architecture']}")
        print(f"   Configuration: {overall_best['config_index']}")
        print(f"   Test Accuracy: {overall_best['test_accuracy']:.4f}")
        print(f"   Validation Accuracy: {overall_best['best_val_accuracy']:.4f}")
        print(f"   Training Time: {overall_best['training_time_seconds']:.1f}s")
        
        print(f"\n🎯 BEST PER ARCHITECTURE:")
        for arch_name, best_model in self.best_model_per_architecture.items():
            print(f"   {arch_name:20s}: {best_model['test_accuracy']:.4f} accuracy")
        
        print(f"\n💾 Models and reports saved to: {self.model_output_dir}")

def main():
    """Main hyperparameter tuning function"""
    # Import project utilities
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from utils import get_data_directories
    
    # Get clean directory paths
    dirs = get_data_directories()
    dataset_path = os.path.join(dirs['labelled_to_process'], "combined_labeled_dataset.csv")
    model_output_dir = dirs['model_output']
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        print("Please run the labelling process first.")
        return
    
    # Initialize trainer
    trainer = HyperparameterTuningTrainer(dataset_path, model_output_dir)
    
    # Run hyperparameter tuning
    trainer.run_hyperparameter_tuning()

if __name__ == "__main__":
    main()