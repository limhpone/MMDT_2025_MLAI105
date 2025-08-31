#!/usr/bin/env python3
"""
Utility script to intelligently copy the best training model to the final model directory
for production use in Streamlit app. Only copies if the new model is better than existing.
"""

import os
import shutil
import glob
import json
from datetime import datetime
from utils import get_data_directories

def get_model_performance(session_pattern):
    """Get performance metrics from a training session"""
    session_report_path = os.path.join(session_pattern, 'training_report.json')
    if os.path.exists(session_report_path):
        try:
            with open(session_report_path, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
            
            performance = report_data.get('performance_analysis', {}).get('overall_performance', {})
            accuracy = performance.get('accuracy', 0)
            rating = report_data.get('performance_analysis', {}).get('performance_summary', {}).get('overall_rating', 'Unknown')
            
            return accuracy, rating
        except Exception as e:
            print(f"⚠️ Could not load performance metrics: {e}")
            return 0, 'Unknown'
    return 0, 'Unknown'

def evaluate_model_performance(model_path, tokenizer_path, params_path):
    """Evaluate the actual performance of an existing model"""
    print(f"   🧪 Evaluating model performance on test data...")
    
    try:
        import pickle
        import numpy as np
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from sklearn.metrics import accuracy_score, classification_report
        
        # Load model and artifacts
        model = load_model(model_path)
        
        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
            
        with open(params_path, 'rb') as f:
            model_params = pickle.load(f)
        
        # Get test data from the labelled dataset
        from utils import get_data_directories
        dirs = get_data_directories()
        dataset_path = os.path.join(dirs['labelled_done'], "combined_labeled_dataset.csv")
        
        if not os.path.exists(dataset_path):
            print(f"   ⚠️ No test dataset found - using estimated accuracy")
            return 0.75, "Manually_Placed_Model", "ESTIMATED_GOOD"
        
        # Load and prepare test data
        import pandas as pd
        df = pd.read_csv(dataset_path)
        
        # Prepare text data
        texts = df['tokens'].tolist()
        labels = df['label'].tolist()
        
        # Convert labels to numeric (labels are already numeric in the dataset)
        numeric_labels = [int(label) for label in labels]
        
        # Tokenize and pad sequences
        sequences = tokenizer.texts_to_sequences(texts)
        X = pad_sequences(sequences, maxlen=model_params['max_length'], padding='post', truncating='post')
        
        # Convert to one-hot encoding
        num_classes = len(model_params['label_mapping'])
        y = tf.keras.utils.to_categorical(numeric_labels, num_classes=num_classes)
        
        # Make predictions with cuDNN fallback handling
        try:
            predictions = model.predict(X, verbose=0)
        except Exception as cudnn_error:
            print(f"   ⚠️ cuDNN error encountered, trying alternative approach...")
            # Try with smaller batch size to avoid cuDNN issues
            try:
                predictions = model.predict(X, verbose=0, batch_size=1)
            except Exception:
                # Last resort: disable cuDNN by recompiling model
                print(f"   🔧 Attempting to disable cuDNN for evaluation...")
                import tensorflow as tf
                # Create a copy of the model with cuDNN disabled
                model_config = model.get_config()
                
                # Modify LSTM layers to disable cuDNN
                for layer_config in model_config['layers']:
                    if layer_config['class_name'] in ['LSTM', 'Bidirectional']:
                        if layer_config['class_name'] == 'Bidirectional':
                            # For bidirectional, modify the wrapped layer
                            layer_config['config']['layer']['config']['use_cudnn'] = False
                        else:
                            # For regular LSTM
                            layer_config['config']['use_cudnn'] = False
                
                # Recreate model without cuDNN
                temp_model = tf.keras.Model.from_config(model_config)
                temp_model.set_weights(model.get_weights())
                
                predictions = temp_model.predict(X, verbose=0)
                print(f"   ✅ Successfully evaluated with cuDNN disabled")
        predicted_labels = np.argmax(predictions, axis=1)
        true_labels = np.argmax(y, axis=1)
        
        # Calculate accuracy
        accuracy = accuracy_score(true_labels, predicted_labels)
        
        # Get classification report
        target_names = list(model_params['label_mapping'].values())
        report = classification_report(true_labels, predicted_labels, target_names=target_names, output_dict=True, zero_division=0)
        
        print(f"   📊 Evaluation Results:")
        print(f"      Test Accuracy: {accuracy:.4f}")
        print(f"      Test Samples: {len(texts)}")
        
        # Determine quality based on accuracy
        if accuracy >= 0.85:
            quality = "EXCELLENT"
        elif accuracy >= 0.70:
            quality = "GOOD"
        elif accuracy >= 0.50:
            quality = "FAIR"
        else:
            quality = "POOR"
        
        return accuracy, "Evaluated_Model", quality
        
    except Exception as e:
        print(f"   ❌ Evaluation failed: {e}")
        print(f"   ⚠️ Falling back to estimated accuracy")
        return 0.75, "Manually_Placed_Model", "ESTIMATED_GOOD"

def analyze_existing_model(final_dir):
    """Analyze an existing model in the final directory to get actual performance"""
    print(f"   🔍 Analyzing existing model files...")
    
    # Check if we have the required files
    model_path = os.path.join(final_dir, 'bilstm_model.h5')
    tokenizer_path = os.path.join(final_dir, 'tokenizer.pickle')
    params_path = os.path.join(final_dir, 'model_params.pickle')
    
    if not all(os.path.exists(f) for f in [model_path, tokenizer_path, params_path]):
        print(f"   ❌ Incomplete model files found")
        return 0, "Incomplete", "UNKNOWN"
    
    # Try to get model info from params first
    try:
        import pickle
        with open(params_path, 'rb') as f:
            model_params = pickle.load(f)
        
        # Extract basic info
        vocab_size = model_params.get('vocab_size', 'Unknown')
        max_length = model_params.get('max_length', 'Unknown')
        label_mapping = model_params.get('label_mapping', {})
        
        print(f"   📊 Model Parameters:")
        print(f"      Vocabulary Size: {vocab_size}")
        print(f"      Max Sequence Length: {max_length}")
        print(f"      Labels: {list(label_mapping.values())}")
        
    except Exception as e:
        print(f"   ❌ Could not read model parameters: {e}")
        return 0, "Unknown", "UNKNOWN"
    
    # Now evaluate the actual performance
    accuracy, model_name, quality = evaluate_model_performance(model_path, tokenizer_path, params_path)
    
    return accuracy, model_name, quality

def display_performance_assessment(accuracy, model_name):
    """Display performance assessment for a model"""
    print(f"   📊 Model: {model_name}")
    print(f"   📈 Accuracy: {accuracy:.4f}")
    
    if isinstance(accuracy, (int, float)):
        if accuracy >= 0.85:
            print(f"   ⭐ Quality: EXCELLENT (≥85%)")
            return "EXCELLENT"
        elif accuracy >= 0.70:
            print(f"   ✅ Quality: GOOD (≥70%)")
            return "GOOD"
        elif accuracy >= 0.50:
            print(f"   ⚠️ Quality: FAIR (≥50%)")
            return "FAIR"
        else:
            print(f"   ❌ Quality: POOR (<50%)")
            return "POOR"
    return "UNKNOWN"

def copy_best_model_to_final():
    """Intelligently copy the best model to the final directory (smart comparison)"""
    
    # Get directories
    dirs = get_data_directories()
    training_dir = dirs['model_output']
    final_dir = dirs['final_model']
    
    print("🔄 Smart Model Deployment System")
    print("=" * 50)
    print(f"📁 Training directory: {training_dir}")
    print(f"📁 Final directory: {final_dir}")
    
    # Find all training models
    model_pattern = os.path.join(training_dir, 'bilstm_model_*.h5')
    model_files = glob.glob(model_pattern)
    
    if not model_files:
        print("❌ No training models found!")
        return False
    
    # Sort by timestamp (newest first)
    model_files.sort(reverse=True)
    latest_model = model_files[0]
    model_name = os.path.basename(latest_model)
    
    # Extract timestamp to find matching session
    timestamp = model_name.replace('bilstm_model_', '').replace('.h5', '')
    
    # Find matching session directory
    training_reports_dir = os.path.join(training_dir, 'training_reports')
    session_pattern = os.path.join(training_reports_dir, f'training_report_{timestamp}')
    
    if not os.path.exists(session_pattern):
        print(f"❌ No matching session directory found: {session_pattern}")
        return False
    
    # Get latest model performance
    latest_accuracy, latest_rating = get_model_performance(session_pattern)
    
    print(f"\n🤖 Latest Training Model Analysis:")
    latest_quality = display_performance_assessment(latest_accuracy, model_name)
    
    # Check if final model directory has an existing model
    final_model_path = os.path.join(final_dir, 'bilstm_model.h5')
    final_tokenizer_path = os.path.join(final_dir, 'tokenizer.pickle')
    final_params_path = os.path.join(final_dir, 'model_params.pickle')
    backup_info_path = os.path.join(final_dir, 'backup_info.json')
    
    # Check if we have a complete model (all required files)
    has_complete_model = (os.path.exists(final_model_path) and 
                         os.path.exists(final_tokenizer_path) and 
                         os.path.exists(final_params_path))
    
    should_copy = False
    copy_reason = ""
    
    if has_complete_model:
        print(f"\n📊 Current Final Model Analysis:")
        
        # Try to get performance from backup_info.json first
        current_accuracy = 0
        current_model_name = "Unknown"
        current_quality = "UNKNOWN"
        
        if os.path.exists(backup_info_path):
            try:
                with open(backup_info_path, 'r', encoding='utf-8') as f:
                    current_backup_info = json.load(f)
                current_accuracy = current_backup_info.get('performance', {}).get('accuracy', 0)
                current_model_name = current_backup_info.get('source_model', 'Unknown')
                current_quality = current_backup_info.get('performance', {}).get('quality', 'UNKNOWN')
                
                print(f"   📊 Found backup info - using stored performance")
                current_quality = display_performance_assessment(current_accuracy, current_model_name)
            except Exception as e:
                print(f"⚠️ Could not read backup info: {e}")
                print(f"   🔍 Will analyze model files directly...")
                current_accuracy, current_model_name, current_quality = analyze_existing_model(final_dir)
        else:
            print(f"   📂 No backup info found - analyzing model files...")
            current_accuracy, current_model_name, current_quality = analyze_existing_model(final_dir)
        
        # Compare models
        print(f"\n⚖️ Model Comparison:")
        print(f"   Current Final: {current_accuracy:.4f} ({current_quality})")
        print(f"   Latest Training: {latest_accuracy:.4f} ({latest_quality})")
        
        if latest_accuracy > current_accuracy:
            should_copy = True
            improvement = ((latest_accuracy - current_accuracy) / current_accuracy) * 100 if current_accuracy > 0 else float('inf')
            copy_reason = f"🚀 UPGRADE: New model is {improvement:.1f}% better!"
        elif latest_accuracy == current_accuracy:
            should_copy = True
            copy_reason = f"🔄 UPDATE: Same performance, updating to latest version"
        else:
            should_copy = False
            decline = ((current_accuracy - latest_accuracy) / current_accuracy) * 100
            copy_reason = f"⛔ SKIP: New model is {decline:.1f}% worse than current"
    else:
        print(f"\n📂 Final model directory is empty or incomplete")
        should_copy = True
        copy_reason = "🎯 INITIAL: No existing model, deploying latest"
    
    print(f"\n🎯 Decision: {copy_reason}")
    
    if not should_copy:
        print(f"✋ Keeping current model - no deployment needed")
        return True  # Return True because this is a valid decision
    
    print(f"\n🔄 Deploying model to final directory...")
    
    # Copy model file
    shutil.copy2(latest_model, final_model_path)
    print(f"✅ Model copied: {final_model_path}")
    
    # Copy tokenizer
    session_tokenizer = os.path.join(session_pattern, 'tokenizer.pickle')
    final_tokenizer = os.path.join(final_dir, 'tokenizer.pickle')
    if os.path.exists(session_tokenizer):
        shutil.copy2(session_tokenizer, final_tokenizer)
        print(f"✅ Tokenizer copied: {final_tokenizer}")
    else:
        print(f"⚠️ Tokenizer not found in session: {session_tokenizer}")
    
    # Copy model parameters
    session_params = os.path.join(session_pattern, 'model_params.pickle')
    final_params = os.path.join(final_dir, 'model_params.pickle')
    if os.path.exists(session_params):
        shutil.copy2(session_params, final_params)
        print(f"✅ Model parameters copied: {final_params}")
    else:
        print(f"⚠️ Model parameters not found in session: {session_params}")
    
    # Create enhanced backup info file
    backup_info = {
        'source_model': model_name,
        'source_session': f'training_report_{timestamp}',
        'copied_at': datetime.now().isoformat(),
        'source_path': latest_model,
        'performance': {
            'accuracy': latest_accuracy,
            'rating': latest_rating,
            'quality': latest_quality
        },
        'deployment_reason': copy_reason
    }
    
    backup_info_path = os.path.join(final_dir, 'backup_info.json')
    with open(backup_info_path, 'w', encoding='utf-8') as f:
        json.dump(backup_info, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Backup info saved: {backup_info_path}")
    
    print(f"\n🎉 Successfully deployed model to final directory!")
    print(f"📊 Model: {model_name}")
    print(f"📁 Session: training_report_{timestamp}")
    print(f"📅 Deployed at: {backup_info['copied_at']}")
    print(f"📈 Performance: {latest_accuracy:.4f} ({latest_quality})")
    
    return True

if __name__ == "__main__":
    try:
        success = copy_best_model_to_final()
        if success:
            print("\n✅ Final model directory is ready for Streamlit app!")
        else:
            print("\n❌ Failed to deploy model to final directory!")
    except Exception as e:
        print(f"\n❌ Error: {e}")