#!/usr/bin/env python3
"""
Utility script to copy the best training model to the final model directory
for production use in Streamlit app.
"""

import os
import shutil
import glob
from datetime import datetime
from utils import get_data_directories

def copy_best_model_to_final():
    """Copy the latest training model to the final model directory"""
    
    # Get directories
    dirs = get_data_directories()
    training_dir = dirs['model_output']
    final_dir = dirs['final_model']
    
    print("🔄 Copying best model to final model directory...")
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
    
    print(f"🤖 Latest training model: {model_name}")
    
    # Extract timestamp to find matching session
    timestamp = model_name.replace('bilstm_model_', '').replace('.h5', '')
    
    # Find matching session directory
    training_reports_dir = os.path.join(training_dir, 'training_reports')
    session_pattern = os.path.join(training_reports_dir, f'training_report_{timestamp}')
    
    if not os.path.exists(session_pattern):
        print(f"❌ No matching session directory found: {session_pattern}")
        return False
    
    print(f"📁 Matching session: training_report_{timestamp}")
    
    # Copy model file
    final_model_path = os.path.join(final_dir, 'bilstm_model.h5')
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
    
    # Create a backup info file
    backup_info = {
        'source_model': model_name,
        'source_session': f'training_report_{timestamp}',
        'copied_at': datetime.now().isoformat(),
        'source_path': latest_model
    }
    
    import json
    backup_info_path = os.path.join(final_dir, 'backup_info.json')
    with open(backup_info_path, 'w', encoding='utf-8') as f:
        json.dump(backup_info, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Backup info saved: {backup_info_path}")
    print(f"\n🎉 Successfully copied best model to final directory!")
    print(f"📊 Model: {model_name}")
    print(f"📁 Session: training_report_{timestamp}")
    print(f"📅 Copied at: {backup_info['copied_at']}")
    
    return True

if __name__ == "__main__":
    try:
        success = copy_best_model_to_final()
        if success:
            print("\n✅ Final model directory is ready for Streamlit app!")
        else:
            print("\n❌ Failed to copy model to final directory!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
