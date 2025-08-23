import os
import pandas as pd
import numpy as np
import pickle
import json
import shutil
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns

class BiLSTMTrainer:
    def __init__(self, dataset_path, model_output_dir):
        """
        Initialize BiLSTM trainer
        
        Args:
            dataset_path: Path to the labeled dataset CSV
            model_output_dir: Directory to save trained model and artifacts
        """
        self.dataset_path = dataset_path
        self.model_output_dir = model_output_dir
        self.model = None
        self.tokenizer = None
        self.max_length = None
        self.vocab_size = None
        
        # Create output directory
        os.makedirs(model_output_dir, exist_ok=True)
        
        # Label mapping
        self.label_mapping = {0: 'red', 1: 'neutral', 2: 'green'}
        
        # Training report data
        self.training_report = {
            'start_time': datetime.now().isoformat(),
            'dataset_info': {},
            'model_config': {},
            'model_architecture': {},
            'training_history': {},
            'evaluation_results': {},
            'confusion_matrix_data': {},
            'performance_analysis': {},
            'end_time': None,
            'total_training_time': None
        }
        
    def load_and_prepare_data(self):
        """Load and prepare the dataset"""
        print("Loading dataset...")
        
        # Load dataset
        df = pd.read_csv(self.dataset_path)
        label_dist = df['category'].value_counts()
        
        print(f"Dataset shape: {df.shape}")
        print(f"Label distribution:\n{label_dist}")
        
        # Store dataset info for report
        self.training_report['dataset_info'] = {
            'dataset_path': self.dataset_path,
            'total_samples': len(df),
            'features': list(df.columns),
            'label_distribution': {
                'red': int(label_dist.get('red', 0)),
                'neutral': int(label_dist.get('neutral', 0)), 
                'green': int(label_dist.get('green', 0))
            },
            'data_balance_ratio': {
                'red': round(label_dist.get('red', 0) / len(df), 3),
                'neutral': round(label_dist.get('neutral', 0) / len(df), 3),
                'green': round(label_dist.get('green', 0) / len(df), 3)
            }
        }
        
        # Prepare features and labels
        texts = df['tokens'].values
        labels = df['label'].values
        
        print(f"Number of texts: {len(texts)}")
        print(f"Number of labels: {len(labels)}")
        
        return texts, labels
    
    def preprocess_texts(self, texts, labels, vocab_size=10000, max_length=100):
        """
        Preprocess texts for training
        
        Args:
            texts: Array of tokenized text strings
            labels: Array of labels
            vocab_size: Maximum vocabulary size
            max_length: Maximum sequence length
        """
        print("Preprocessing texts...")
        
        # Initialize tokenizer
        self.tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
        self.tokenizer.fit_on_texts(texts)
        
        # Convert texts to sequences
        sequences = self.tokenizer.texts_to_sequences(texts)
        
        # Pad sequences
        padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
        
        # Convert labels to categorical
        categorical_labels = to_categorical(labels, num_classes=3)
        
        # Store parameters
        self.vocab_size = min(vocab_size, len(self.tokenizer.word_index)) + 1
        self.max_length = max_length
        
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Max sequence length: {self.max_length}")
        print(f"Padded sequences shape: {padded_sequences.shape}")
        print(f"Categorical labels shape: {categorical_labels.shape}")
        
        return padded_sequences, categorical_labels
    
    def build_model(self, embedding_dim=100, lstm_units=64, dropout_rate=0.3):
        """
        Build Bi-LSTM model architecture
        
        Args:
            embedding_dim: Dimension of word embeddings
            lstm_units: Number of LSTM units
            dropout_rate: Dropout rate for regularization
        """
        # Store model configuration for report
        self.training_report['model_config'] = {
            'architecture': 'Bidirectional LSTM',
            'embedding_dim': embedding_dim,
            'lstm_units': lstm_units,
            'dropout_rate': dropout_rate,
            'vocab_size': self.vocab_size,
            'max_sequence_length': self.max_length,
            'num_classes': 3,
            'optimizer': 'adam',
            'loss_function': 'categorical_crossentropy'
        }
        print("Building Bi-LSTM model...")
        
        model = Sequential([
            # Embedding layer
            Embedding(
                input_dim=self.vocab_size,
                output_dim=embedding_dim,
                input_length=self.max_length,
                mask_zero=True
            ),
            
            # Bidirectional LSTM layers
            Bidirectional(LSTM(lstm_units, return_sequences=True, dropout=dropout_rate)),
            Bidirectional(LSTM(lstm_units, dropout=dropout_rate)),
            
            # Dense layers
            Dense(64, activation='relu'),
            Dropout(dropout_rate),
            Dense(32, activation='relu'),
            Dropout(dropout_rate),
            
            # Output layer
            Dense(3, activation='softmax')  # 3 classes: red, neutral, green
        ])
        
        self.model = model
        
        # Print model summary
        print("Model architecture:")
        model.summary()
        
        # Capture model architecture details for report
        self._capture_model_architecture(model)
        
        return model
    
    def _capture_model_architecture(self, model):
        """Capture detailed model architecture information"""
        import io
        
        # First compile the model to ensure it's built properly
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        # Capture model summary as string
        stringio = io.StringIO()
        model.summary(print_fn=lambda x: stringio.write(x + '\n'))
        model_summary_text = stringio.getvalue()
        stringio.close()
        
        # Get layer details safely
        layers_info = []
        
        for i, layer in enumerate(model.layers):
            try:
                # Try to get parameter count, but handle unbuilt layers gracefully
                try:
                    param_count = layer.count_params()
                except (ValueError, AttributeError):
                    # If layer isn't built or doesn't have parameters, set to 0
                    param_count = 0
                
                layer_info = {
                    'layer_index': i,
                    'layer_name': layer.name,
                    'layer_type': layer.__class__.__name__,
                    'output_shape': str(layer.output_shape) if hasattr(layer, 'output_shape') else 'Not Built',
                    'param_count': param_count,
                    'trainable': layer.trainable
                }
                
                # Add layer-specific details safely
                if hasattr(layer, 'units') and layer.units:
                    layer_info['units'] = layer.units
                if hasattr(layer, 'activation') and layer.activation:
                    layer_info['activation'] = str(layer.activation.__name__)
                if hasattr(layer, 'rate'):  # For dropout layers
                    layer_info['dropout'] = layer.rate
                if hasattr(layer, 'input_dim') and layer.input_dim:
                    layer_info['input_dim'] = layer.input_dim
                if hasattr(layer, 'output_dim') and layer.output_dim:
                    layer_info['output_dim'] = layer.output_dim
                
                layers_info.append(layer_info)
                
            except Exception as e:
                # If any error occurs with a layer, create a basic entry
                layer_info = {
                    'layer_index': i,
                    'layer_name': getattr(layer, 'name', f'layer_{i}'),
                    'layer_type': layer.__class__.__name__,
                    'output_shape': 'Error',
                    'param_count': 0,
                    'trainable': getattr(layer, 'trainable', True)
                }
                layers_info.append(layer_info)
                print(f"Warning: Could not get details for layer {i}: {e}")
        
        # Get total parameters from model (more reliable)
        try:
            total_params = model.count_params()
            # Try to distinguish trainable vs non-trainable
            trainable_params = sum(layer.count_params() for layer in model.layers if layer.trainable and hasattr(layer, 'count_params'))
        except:
            # Fallback calculation
            total_params = sum(layer_info['param_count'] for layer_info in layers_info)
            trainable_params = sum(layer_info['param_count'] for layer_info in layers_info if layer_info['trainable'])
        
        # Store architecture information
        self.training_report['model_architecture'] = {
            'model_summary': model_summary_text,
            'total_layers': len(model.layers),
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params,
            'layers_details': layers_info,
            'input_shape': str(getattr(model, 'input_shape', 'Unknown')),
            'output_shape': str(getattr(model, 'output_shape', 'Unknown'))
        }
    
    def train_model(self, X, y, test_size=0.2, validation_split=0.2, epochs=10, batch_size=32):
        """
        Train the Bi-LSTM model
        
        Args:
            X: Preprocessed sequences
            y: Categorical labels
            test_size: Proportion of data to use for testing
            validation_split: Proportion of training data to use for validation
            epochs: Number of training epochs
            batch_size: Training batch size
        """
        print("Training model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y.argmax(axis=1)
        )
        
        print(f"Training set shape: {X_train.shape}, {y_train.shape}")
        print(f"Test set shape: {X_test.shape}, {y_test.shape}")
        
        # Define callbacks with improved patience for more training
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=8,  # Increased patience to allow more training
                restore_best_weights=True,
                min_delta=0.001  # Minimum change to qualify as improvement
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,  # Less aggressive reduction
                patience=5,  # More patience before reducing LR
                min_lr=0.0001,  # Lower minimum learning rate
                verbose=1
            )
        ]
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Test accuracy: {test_accuracy:.4f}")
        print(f"Test loss: {test_loss:.4f}")
        
        # Generate predictions for detailed metrics
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        # Classification report
        print("\nClassification Report:")
        target_names = [self.label_mapping[i] for i in range(3)]
        class_report = classification_report(y_true_classes, y_pred_classes, target_names=target_names, zero_division=0)
        print(class_report)
        
        # Store evaluation results for report
        class_report_dict = classification_report(y_true_classes, y_pred_classes, target_names=target_names, output_dict=True, zero_division=0)
        self.training_report['evaluation_results'] = {
            'classification_report_text': class_report,
            'classification_report_dict': class_report_dict,
            'test_accuracy': float(test_accuracy),
            'test_loss': float(test_loss)
        }
        
        # Store confusion matrix data
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        self.training_report['confusion_matrix_data'] = {
            'matrix': cm.tolist(),
            'labels': target_names
        }
        
        # Perform dynamic performance analysis
        self._analyze_model_performance(class_report_dict, cm, target_names)
        
        # Store training history with proper epoch tracking
        self.training_report['training_history'] = {
            'epochs_completed': len(history.history['accuracy']),
            'final_training_accuracy': float(history.history['accuracy'][-1]),
            'final_validation_accuracy': float(history.history['val_accuracy'][-1]),
            'final_training_loss': float(history.history['loss'][-1]),
            'final_validation_loss': float(history.history['val_loss'][-1]),
            'history_data': {
                'accuracy': [float(x) for x in history.history['accuracy']],
                'val_accuracy': [float(x) for x in history.history['val_accuracy']],
                'loss': [float(x) for x in history.history['loss']],
                'val_loss': [float(x) for x in history.history['val_loss']]
            }
        }
        
        # Store test data for later use
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = y_pred
        
        return history
    
    def _analyze_model_performance(self, class_report_dict, confusion_matrix, target_names):
        """Dynamically analyze model performance across all categories"""
        
        # Performance thresholds
        EXCELLENT_THRESHOLD = 0.85
        GOOD_THRESHOLD = 0.70
        POOR_THRESHOLD = 0.50
        CRITICAL_THRESHOLD = 0.10
        
        analysis = {
            'overall_performance': {},
            'class_analysis': {},
            'critical_issues': [],
            'recommendations': [],
            'performance_summary': {}
        }
        
        # Overall performance analysis
        overall_acc = class_report_dict.get('accuracy', 0)
        macro_avg = class_report_dict.get('macro avg', {})
        weighted_avg = class_report_dict.get('weighted avg', {})
        
        analysis['overall_performance'] = {
            'accuracy': overall_acc,
            'macro_precision': macro_avg.get('precision', 0),
            'macro_recall': macro_avg.get('recall', 0),
            'macro_f1': macro_avg.get('f1-score', 0),
            'weighted_precision': weighted_avg.get('precision', 0),
            'weighted_recall': weighted_avg.get('recall', 0),
            'weighted_f1': weighted_avg.get('f1-score', 0)
        }
        
        # Classify overall performance
        if overall_acc >= EXCELLENT_THRESHOLD:
            analysis['overall_performance']['rating'] = 'Excellent'
        elif overall_acc >= GOOD_THRESHOLD:
            analysis['overall_performance']['rating'] = 'Good'
        elif overall_acc >= POOR_THRESHOLD:
            analysis['overall_performance']['rating'] = 'Poor'
        else:
            analysis['overall_performance']['rating'] = 'Critical'
        
        # Per-class analysis
        class_issues = []
        for class_name in target_names:
            if class_name in class_report_dict:
                metrics = class_report_dict[class_name]
                precision = metrics.get('precision', 0)
                recall = metrics.get('recall', 0)
                f1 = metrics.get('f1-score', 0)
                support = metrics.get('support', 0)
                
                class_analysis = {
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'support': support,
                    'issues': [],
                    'rating': 'Good'
                }
                
                # Identify specific issues for this class
                if recall <= CRITICAL_THRESHOLD:
                    class_analysis['issues'].append('Critical recall - model almost never predicts this class')
                    class_analysis['rating'] = 'Critical'
                    class_issues.append(f"{class_name.upper()} has critical recall ({recall:.3f}) - model fails to identify this class")
                elif recall <= POOR_THRESHOLD:
                    class_analysis['issues'].append('Low recall - model often misses this class')
                    class_analysis['rating'] = 'Poor'
                    class_issues.append(f"{class_name.upper()} has poor recall ({recall:.3f})")
                
                if precision <= CRITICAL_THRESHOLD:
                    class_analysis['issues'].append('Critical precision - model predictions are unreliable')
                    class_analysis['rating'] = 'Critical'
                    class_issues.append(f"{class_name.upper()} has critical precision ({precision:.3f}) - predictions are unreliable")
                elif precision <= POOR_THRESHOLD:
                    class_analysis['issues'].append('Low precision - many false positives')
                    if class_analysis['rating'] != 'Critical':
                        class_analysis['rating'] = 'Poor'
                    class_issues.append(f"{class_name.upper()} has poor precision ({precision:.3f})")
                
                if f1 <= CRITICAL_THRESHOLD:
                    class_analysis['issues'].append('Critical F1-score - overall poor performance')
                    class_analysis['rating'] = 'Critical'
                elif f1 <= POOR_THRESHOLD:
                    class_analysis['issues'].append('Low F1-score - needs improvement')
                    if class_analysis['rating'] not in ['Critical']:
                        class_analysis['rating'] = 'Poor'
                
                # Check for class imbalance effects
                total_support = sum(class_report_dict[cls].get('support', 0) for cls in target_names if cls in class_report_dict)
                class_ratio = support / total_support if total_support > 0 else 0
                
                if class_ratio < 0.2:  # Less than 20% of data
                    class_analysis['issues'].append('Minority class - may need oversampling or class weights')
                    class_issues.append(f"{class_name.upper()} is a minority class ({class_ratio:.1%} of data)")
                
                analysis['class_analysis'][class_name] = class_analysis
        
        # Generate recommendations based on analysis
        recommendations = []
        
        if overall_acc < GOOD_THRESHOLD:
            recommendations.append("Overall accuracy is below 70% - consider model architecture improvements")
        
        if class_issues:
            analysis['critical_issues'] = class_issues
            
            # Check for severe class imbalance
            class_supports = [class_report_dict[cls].get('support', 0) for cls in target_names if cls in class_report_dict]
            if class_supports:
                max_support = max(class_supports)
                min_support = min(class_supports)
                imbalance_ratio = max_support / min_support if min_support > 0 else float('inf')
                
                if imbalance_ratio > 3:
                    recommendations.extend([
                        "Severe class imbalance detected - use class_weight='balanced' in model.fit()",
                        "Consider SMOTE or other oversampling techniques for minority classes",
                        "Collect more data for underrepresented classes"
                    ])
                
            # Check for specific class issues
            critical_classes = [name for name, analysis_data in analysis['class_analysis'].items() 
                              if analysis_data['rating'] == 'Critical']
            
            if critical_classes:
                recommendations.extend([
                    f"Critical performance issues in: {', '.join(critical_classes)}",
                    "Consider adjusting decision thresholds for problematic classes",
                    "Feature engineering may be needed for poorly performing classes",
                    "Try ensemble methods or different model architectures"
                ])
        
        # Check confusion matrix for specific patterns
        cm = confusion_matrix
        for i, true_class in enumerate(target_names):
            for j, pred_class in enumerate(target_names):
                if i != j and cm[i][j] > cm[i][i]:  # More misclassified than correct
                    recommendations.append(f"{true_class.upper()} is often confused with {pred_class.upper()} - check feature discrimination")
        
        analysis['recommendations'] = recommendations
        
        # Performance summary
        critical_count = sum(1 for cls_data in analysis['class_analysis'].values() if cls_data['rating'] == 'Critical')
        poor_count = sum(1 for cls_data in analysis['class_analysis'].values() if cls_data['rating'] == 'Poor')
        good_count = len(analysis['class_analysis']) - critical_count - poor_count
        
        analysis['performance_summary'] = {
            'total_classes': len(target_names),
            'critical_classes': critical_count,
            'poor_classes': poor_count,
            'good_classes': good_count,
            'needs_immediate_attention': critical_count > 0,
            'overall_rating': analysis['overall_performance']['rating']
        }
        
        self.training_report['performance_analysis'] = analysis
    
    def plot_training_history(self, history):
        """Plot training history"""
        print("Plotting training history...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy plot
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Loss plot
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        
        # Save plot to session reports directory
        plot_path = os.path.join(self.session_reports_dir, 'training_history.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        print(f"Training history plot saved to: {plot_path}")
        plt.show()
    
    def plot_confusion_matrix(self):
        """Plot confusion matrix"""
        print("Plotting confusion matrix...")
        
        y_pred_classes = np.argmax(self.y_pred, axis=1)
        y_true_classes = np.argmax(self.y_test, axis=1)
        
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        
        plt.figure(figsize=(8, 6))
        target_names = [self.label_mapping[i] for i in range(3)]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=target_names, yticklabels=target_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # Save plot to session reports directory
        plot_path = os.path.join(self.session_reports_dir, 'confusion_matrix.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        print(f"Confusion matrix saved to: {plot_path}")
        plt.show()
    
    def save_model_and_artifacts(self):
        """Save trained model and preprocessing artifacts"""
        print("Saving model and artifacts...")
        
        # Extract timestamp from the existing session directory
        session_dir_name = os.path.basename(self.session_reports_dir)
        timestamp = session_dir_name.replace('training_report_', '')
        session_name = f"training_{timestamp}"
        
        # Save model with session name
        model_filename = f'bilstm_model_{timestamp}.h5'
        model_path = os.path.join(self.model_output_dir, model_filename)
        self.model.save(model_path)
        print(f"Model saved to: {model_path}")
        
        # Save tokenizer to session reports directory
        tokenizer_path = os.path.join(self.session_reports_dir, 'tokenizer.pickle')
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)
        print(f"Tokenizer saved to: {tokenizer_path}")
        
        # Save model parameters to session reports directory
        params = {
            'vocab_size': self.vocab_size,
            'max_length': self.max_length,
            'label_mapping': self.label_mapping
        }
        params_path = os.path.join(self.session_reports_dir, 'model_params.pickle')
        with open(params_path, 'wb') as f:
            pickle.dump(params, f)
        print(f"Model parameters saved to: {params_path}")
        
        # Save session info to session reports directory
        session_info = {
            'session_name': session_name,
            'timestamp': timestamp,
            'model_filename': model_filename,
            'training_date': datetime.now().isoformat()
        }
        session_info_path = os.path.join(self.session_reports_dir, 'session_info.pickle')
        with open(session_info_path, 'wb') as f:
            pickle.dump(session_info, f)
        print(f"Session info saved to: {session_info_path}")
        
        return session_name
    
    def generate_training_report(self, session_name=None):
        """Generate comprehensive training report in markdown format"""
        print("Generating comprehensive training report...")
        
        # Complete the report with end time
        self.training_report['end_time'] = datetime.now().isoformat()
        start_time = datetime.fromisoformat(self.training_report['start_time'])
        end_time = datetime.fromisoformat(self.training_report['end_time'])
        self.training_report['total_training_time'] = str(end_time - start_time)
        
        # Use provided session name or create one
        if session_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            session_name = f"training_{timestamp}"
        
        # Generate markdown report
        report_md = self._generate_markdown_content()
        
        # Save markdown report in session reports directory
        report_path = os.path.join(self.session_reports_dir, 'training_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_md)
        
        # Save JSON report for programmatic access
        json_report_path = os.path.join(self.session_reports_dir, 'training_report.json')
        with open(json_report_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_report, f, indent=2, ensure_ascii=False)
        
        print(f"📋 Training report saved to: {report_path}")
        print(f"📊 JSON report saved to: {json_report_path}")
        print(f"📁 Reports directory: {self.session_reports_dir}")
        
        return report_path
    
    def _generate_markdown_content(self):
        """Generate markdown content for the training report"""
        report = self.training_report
        
        md_content = f"""# BiLSTM Training Report

Generated on: {report['end_time']}
Training Duration: {report['total_training_time']}

---

## 📊 Dataset Information

| Metric | Value |
|--------|-------|
| **Dataset Path** | `{report['dataset_info']['dataset_path']}` |
| **Total Samples** | {report['dataset_info']['total_samples']:,} |
| **Features** | {', '.join(report['dataset_info']['features'])} |

### Label Distribution

| Class | Count | Percentage |
|-------|-------|------------|
| 🔴 Red | {report['dataset_info']['label_distribution']['red']:,} | {report['dataset_info']['data_balance_ratio']['red']*100:.1f}% |
| ⚪ Neutral | {report['dataset_info']['label_distribution']['neutral']:,} | {report['dataset_info']['data_balance_ratio']['neutral']*100:.1f}% |
| 🟢 Green | {report['dataset_info']['label_distribution']['green']:,} | {report['dataset_info']['data_balance_ratio']['green']*100:.1f}% |

### ⚠️ Data Balance Analysis
"""
        
        # Add data balance warnings
        ratios = report['dataset_info']['data_balance_ratio']
        min_ratio = min(ratios.values())
        max_ratio = max(ratios.values())
        imbalance_ratio = max_ratio / min_ratio if min_ratio > 0 else float('inf')
        
        if imbalance_ratio > 2:
            md_content += f"""
**🚨 CLASS IMBALANCE DETECTED**: Ratio of {imbalance_ratio:.1f}:1 between most and least frequent classes.
This likely explains poor performance on the RED category (minority class).

**Recommendations:**
- Use class weights during training
- Apply SMOTE or other oversampling techniques
- Collect more RED category samples
- Use stratified sampling
"""
        else:
            md_content += "✅ Classes are reasonably balanced.\n"
        
        md_content += f"""

---

## 🏗️ Model Configuration

| Parameter | Value |
|-----------|-------|
| **Architecture** | {report['model_config']['architecture']} |
| **Embedding Dimension** | {report['model_config']['embedding_dim']} |
| **LSTM Units** | {report['model_config']['lstm_units']} |
| **Dropout Rate** | {report['model_config']['dropout_rate']} |
| **Vocabulary Size** | {report['model_config']['vocab_size']:,} |
| **Max Sequence Length** | {report['model_config']['max_sequence_length']} |
| **Number of Classes** | {report['model_config']['num_classes']} |
| **Optimizer** | {report['model_config']['optimizer']} |
| **Loss Function** | {report['model_config']['loss_function']} |

## 🔧 Model Architecture Details

### Summary
| Metric | Value |
|--------|-------|
| **Total Layers** | {report['model_architecture']['total_layers']} |
| **Total Parameters** | {report['model_architecture']['total_parameters']:,} |
| **Trainable Parameters** | {report['model_architecture']['trainable_parameters']:,} |
| **Non-trainable Parameters** | {report['model_architecture']['non_trainable_parameters']:,} |
| **Input Shape** | {report['model_architecture']['input_shape']} |
| **Output Shape** | {report['model_architecture']['output_shape']} |

### Layer-by-Layer Structure

| Layer | Type | Output Shape | Parameters | Details |
|-------|------|--------------|------------|---------|"""
        
        # Add layer details
        for layer in report['model_architecture']['layers_details']:
            details = []
            if 'units' in layer:
                details.append(f"units={layer['units']}")
            if 'activation' in layer:
                details.append(f"activation={layer['activation']}")
            if 'dropout' in layer:
                details.append(f"dropout={layer['dropout']}")
            details_str = ", ".join(details) if details else "N/A"
            
            md_content += f"\n| {layer['layer_index']} | {layer['layer_type']} | {layer['output_shape']} | {layer['param_count']:,} | {details_str} |"
        
        md_content += f"""

### Complete Model Summary
```
{report['model_architecture']['model_summary']}
```

---

## 📈 Training History

| Metric | Value |
|--------|-------|
| **Epochs Completed** | {report['training_history']['epochs_completed']} |
| **Final Training Accuracy** | {report['training_history']['final_training_accuracy']:.4f} |
| **Final Validation Accuracy** | {report['training_history']['final_validation_accuracy']:.4f} |
| **Final Training Loss** | {report['training_history']['final_training_loss']:.4f} |
| **Final Validation Loss** | {report['training_history']['final_validation_loss']:.4f} |

### Training Progress by Epoch

| Epoch | Train Acc | Val Acc | Train Loss | Val Loss |
|-------|-----------|---------|------------|----------|"""
        
        # Add epoch-by-epoch data
        history_data = report['training_history']['history_data']
        for i in range(len(history_data['accuracy'])):
            epoch = i + 1
            train_acc = history_data['accuracy'][i]
            val_acc = history_data['val_accuracy'][i]
            train_loss = history_data['loss'][i]
            val_loss = history_data['val_loss'][i]
            md_content += f"\n| {epoch} | {train_acc:.4f} | {val_acc:.4f} | {train_loss:.4f} | {val_loss:.4f} |"
        
        md_content += f"""

---

## 🎯 Model Evaluation Results

### Test Set Performance
| Metric | Value |
|--------|-------|
| **Test Accuracy** | {report['evaluation_results']['test_accuracy']:.4f} |
| **Test Loss** | {report['evaluation_results']['test_loss']:.4f} |

### Classification Report
```
{report['evaluation_results']['classification_report_text']}
```

### Detailed Performance by Class

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|"""
        
        # Add per-class metrics
        class_report = report['evaluation_results']['classification_report_dict']
        for class_name in ['red', 'neutral', 'green']:
            if class_name in class_report:
                metrics = class_report[class_name]
                md_content += f"\n| {class_name.title()} | {metrics['precision']:.2f} | {metrics['recall']:.2f} | {metrics['f1-score']:.2f} | {int(metrics['support'])} |"
        
        # Add confusion matrix
        cm = report['confusion_matrix_data']['matrix']
        labels = report['confusion_matrix_data']['labels']
        
        md_content += f"""

### Confusion Matrix

|  | Predicted Red | Predicted Neutral | Predicted Green |
|--|---------------|-------------------|-----------------|
| **Actual Red** | {cm[0][0]} | {cm[0][1]} | {cm[0][2]} |
| **Actual Neutral** | {cm[1][0]} | {cm[1][1]} | {cm[1][2]} |
| **Actual Green** | {cm[2][0]} | {cm[2][1]} | {cm[2][2]} |

---

## 🔍 Dynamic Performance Analysis

### Overall Performance Rating: {report['performance_analysis']['performance_summary']['overall_rating']}

| Class Performance Summary | Count |
|---------------------------|-------|
| 🟢 **Good Classes** | {report['performance_analysis']['performance_summary']['good_classes']} |
| 🟡 **Poor Classes** | {report['performance_analysis']['performance_summary']['poor_classes']} |
| 🔴 **Critical Classes** | {report['performance_analysis']['performance_summary']['critical_classes']} |
| ⚠️ **Needs Immediate Attention** | {"Yes" if report['performance_analysis']['performance_summary']['needs_immediate_attention'] else "No"} |

### Per-Class Analysis"""
        
        # Add dynamic per-class analysis
        for class_name, class_data in report['performance_analysis']['class_analysis'].items():
            rating_emoji = {'Excellent': '🟢', 'Good': '🟢', 'Poor': '🟡', 'Critical': '🔴'}.get(class_data['rating'], '⚪')
            
            md_content += f"""

#### {rating_emoji} {class_name.upper()} Class - Rating: {class_data['rating']}
| Metric | Value |
|--------|-------|
| **Precision** | {class_data['precision']:.3f} |
| **Recall** | {class_data['recall']:.3f} |
| **F1-Score** | {class_data['f1_score']:.3f} |
| **Support** | {class_data['support']} |

"""
            if class_data['issues']:
                md_content += "**Issues Identified:**\n"
                for issue in class_data['issues']:
                    md_content += f"- {issue}\n"
        
        # Add critical issues section
        if report['performance_analysis']['critical_issues']:
            md_content += f"""

### 🚨 Critical Issues Detected

{len(report['performance_analysis']['critical_issues'])} critical issues found:

"""
            for issue in report['performance_analysis']['critical_issues']:
                md_content += f"- **{issue}**\n"
        else:
            md_content += "\n### ✅ No Critical Issues Detected\n"
        
        # Add recommendations
        if report['performance_analysis']['recommendations']:
            md_content += f"""

### 💡 Automated Recommendations

{len(report['performance_analysis']['recommendations'])} recommendations to improve model performance:

"""
            for i, rec in enumerate(report['performance_analysis']['recommendations'], 1):
                md_content += f"{i}. {rec}\n"
        
        md_content += f"""

### Model Artifacts Generated
- 📊 `training_history.png` - Training curves visualization
- 🎯 `confusion_matrix.png` - Confusion matrix heatmap  
- 🤖 `bilstm_model.h5` - Trained model file
- 🔤 `tokenizer.pickle` - Text tokenizer
- ⚙️ `model_params.pickle` - Model parameters
- 📋 `training_report.json` - Machine-readable report data

---

## 💡 Next Steps

1. **Address Class Imbalance**: Use techniques mentioned above
2. **Hyperparameter Tuning**: Experiment with different model configurations
3. **Data Augmentation**: Generate more diverse training samples
4. **Feature Engineering**: Add Myanmar-specific linguistic features
5. **Ensemble Methods**: Combine multiple models for better performance
6. **Cross-Validation**: Validate results with k-fold cross-validation

---

*Report generated by BiLSTM Trainer v1.0*
"""
        
        return md_content
    
    def move_processed_files(self):
        """Move processed files from to_process to done"""
        print("Moving processed files to done folder...")
        
        # Import project utilities
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
        from utils import get_data_directories
        
        dirs = get_data_directories()
        to_process_dir = dirs['labelled_to_process']
        done_dir = dirs['labelled_done']
        
        # Create done directory
        os.makedirs(done_dir, exist_ok=True)
        
        if not os.path.exists(to_process_dir):
            print(f"To_process directory doesn't exist: {to_process_dir}")
            return
        
        # Move all files from to_process to done
        files_moved = 0
        for filename in os.listdir(to_process_dir):
            if filename.endswith('.csv'):
                src = os.path.join(to_process_dir, filename)
                dst = os.path.join(done_dir, filename)
                try:
                    import shutil
                    shutil.move(src, dst)
                    files_moved += 1
                    print(f"  Moved {filename} to labelled/done/")
                except Exception as e:
                    print(f"  Error moving {filename}: {e}")
        
        print(f"Moved {files_moved} files to done folder")
    
    def _auto_copy_to_final(self):
        """Automatically run smart model deployment to final directory"""
        try:
            # Import the copy function from the utility script
            import sys
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
            from copy_best_model import copy_best_model_to_final
            
            success = copy_best_model_to_final()
            if success:
                print("🎯 Smart deployment completed!")
                print("🚀 Check final model directory status above")
            else:
                print("❌ Smart deployment failed")
                print("   Please run manually: python copy_best_model.py")
                
        except Exception as e:
            print(f"❌ Error during smart deployment: {e}")
            print("   Please run manually: python copy_best_model.py")
    
    def run_full_training(self, vocab_size=10000, max_length=100, embedding_dim=100,
                         lstm_units=64, dropout_rate=0.3, epochs=10, batch_size=32):
        """
        Run the complete training pipeline
        
        Args:
            vocab_size: Maximum vocabulary size
            max_length: Maximum sequence length
            embedding_dim: Dimension of word embeddings
            lstm_units: Number of LSTM units
            dropout_rate: Dropout rate
            epochs: Number of training epochs
            batch_size: Training batch size
        """
        print("=== Starting Bi-LSTM Training Pipeline ===")
        
        # Create training reports directory structure
        training_reports_dir = os.path.join(self.model_output_dir, 'training_reports')
        os.makedirs(training_reports_dir, exist_ok=True)
        
        # We'll create the session directory later when we save the model
        # to ensure the timestamp matches
        self.session_reports_dir = None
        
        print(f"📁 Training reports directory ready: {training_reports_dir}")
        
        # Load data
        texts, labels = self.load_and_prepare_data()
        
        # Preprocess
        X, y = self.preprocess_texts(texts, labels, vocab_size, max_length)
        
        # Build model
        self.build_model(embedding_dim, lstm_units, dropout_rate)
        
        # Train model
        history = self.train_model(X, y, epochs=epochs, batch_size=batch_size)
        
        # Create session directory for plotting
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        training_reports_dir = os.path.join(self.model_output_dir, 'training_reports')
        self.session_reports_dir = os.path.join(training_reports_dir, f"training_report_{timestamp}")
        os.makedirs(self.session_reports_dir, exist_ok=True)
        print(f"📁 Created session directory for plotting: {self.session_reports_dir}")
        
        # Plot results (now they'll save to the session directory)
        self.plot_training_history(history)
        self.plot_confusion_matrix()
        
        # Save everything and get session name
        session_name = self.save_model_and_artifacts()
        
        # Generate comprehensive training report
        self.generate_training_report(session_name)
        
        # Move processed files from to_process to done
        self.move_processed_files()
        
        print("=== Training Complete! ===")
        print(f"📋 Check training reports in: {self.session_reports_dir}")
        print("📋 Check training_report.md for detailed analysis and recommendations!")
        
        # Auto-copy best model to final directory (always update with latest)
        best_val_acc = max(history.history['val_accuracy']) if 'val_accuracy' in history.history else 0
        print(f"\n📊 Model Performance Summary:")
        print(f"   Best validation accuracy: {best_val_acc:.4f}")
        
        # Evaluate performance and provide feedback
        if best_val_acc >= 0.85:
            print(f"   Performance Rating: ⭐ EXCELLENT (≥85%)")
            print(f"   Status: 🎉 Ready for production deployment!")
        elif best_val_acc >= 0.70:
            print(f"   Performance Rating: ✅ GOOD (≥70%)")
            print(f"   Status: 🚀 Suitable for production use")
        elif best_val_acc >= 0.50:
            print(f"   Performance Rating: ⚠️ FAIR (≥50%)")
            print(f"   Status: 🔧 Consider model improvements")
        else:
            print(f"   Performance Rating: ❌ POOR (<50%)")
            print(f"   Status: 🛠️ Needs significant improvements")
        
        print(f"\n🔄 Running smart model deployment...")
        self._auto_copy_to_final()

def main():
    """Main training function"""
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
    trainer = BiLSTMTrainer(dataset_path, model_output_dir)
    
    # Run training with improved parameters based on analysis
    trainer.run_full_training(
        vocab_size=15000,      # Slightly smaller vocab for better generalization
        max_length=500,        # Reduced length for better performance  
        embedding_dim=256,     # Larger embeddings for better representation
        lstm_units=128,        # More units for better capacity
        dropout_rate=0.5,      # Higher dropout to prevent overfitting
        epochs=50,             # More epochs (early stopping will control)
        batch_size=64          # Larger batch size for stable gradients
    )

if __name__ == "__main__":
    main()