#!/usr/bin/env python3
"""
Test script to verify auto-copy functionality
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '3_trainer', 'trainer'))

from bilstm_trainer import BiLSTMTrainer
from utils import get_data_directories

def test_auto_copy():
    """Test the auto-copy functionality"""
    
    print("🧪 Testing Auto-Copy Functionality")
    print("=" * 50)
    
    # Get directories
    dirs = get_data_directories()
    dataset_path = os.path.join(dirs['labelled_to_process'], "combined_labeled_dataset.csv")
    model_output_dir = dirs['model_output']
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        return False
    
    # Initialize trainer
    trainer = BiLSTMTrainer(dataset_path, model_output_dir)
    
    # Test the auto-copy method directly
    print("🔄 Testing _auto_copy_to_final method...")
    try:
        trainer._auto_copy_to_final()
        print("✅ Auto-copy test completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Auto-copy test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_auto_copy()
    if success:
        print("\n🎉 All tests passed! Auto-copy is ready for training pipeline.")
    else:
        print("\n❌ Tests failed. Please check the implementation.")
