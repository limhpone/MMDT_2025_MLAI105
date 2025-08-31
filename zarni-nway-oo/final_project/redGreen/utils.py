#!/usr/bin/env python3
"""
Shared utilities for Myanmar News Classification Project
"""

import os

def find_project_root():
    """
    Find project root by looking for bilstm_pipeline.py marker file
    
    This function traverses up the directory tree from the current script location
    until it finds the directory containing bilstm_pipeline.py, which indicates
    the project root.
    
    Returns:
        str: Absolute path to project root directory
        
    Raises:
        RuntimeError: If project root cannot be found
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Traverse up the directory tree
    while current_dir != os.path.dirname(current_dir):  # Stop at filesystem root
        # Check if this directory contains the marker file
        if os.path.exists(os.path.join(current_dir, 'bilstm_pipeline.py')):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    
    raise RuntimeError("Could not find project root. Make sure bilstm_pipeline.py exists in the project root.")

def get_data_directories():
    """
    Get standard data directories relative to project root
    
    Returns:
        dict: Dictionary containing paths to standard data directories
    """
    project_root = find_project_root()
    
    return {
        'raw_scraped': os.path.join(project_root, 'data', 'raw', 'scraped'),
        'raw_to_process': os.path.join(project_root, 'data', 'raw', 'to_process'),
        'raw_done': os.path.join(project_root, 'data', 'raw', 'done'),
        'cleaned_to_process': os.path.join(project_root, 'data', 'cleaned', 'to_process'),
        'cleaned_done': os.path.join(project_root, 'data', 'cleaned', 'done'),
        'preprocessed_to_process': os.path.join(project_root, 'data', 'preprocessed', 'to_process'),
        'preprocessed_done': os.path.join(project_root, 'data', 'preprocessed', 'done'),
        'tokenized_to_process': os.path.join(project_root, 'data', 'tokenized', 'to_process'),
        'tokenized_done': os.path.join(project_root, 'data', 'tokenized', 'done'),
        'labelled_to_process': os.path.join(project_root, 'data', 'labelled', 'to_process'),
        'labelled_done': os.path.join(project_root, 'data', 'labelled', 'done'),
        'model_tester_raw': os.path.join(project_root, 'data', 'model_tester', 'raw'),
        'model_tester_processed': os.path.join(project_root, 'data', 'model_tester', 'processed'),
        'model_tester_done': os.path.join(project_root, 'data', 'model_tester', 'done'),
        'model_output': os.path.join(project_root, '3_trainer', 'output_model'),
        'final_model': os.path.join(project_root, '00_final_model'),
        'analysis_output': os.path.join(project_root, '4_analyzer', 'analysis_outputs')
    }

# Test the function when run directly
if __name__ == "__main__":
    try:
        root = find_project_root()
        print(f"Project root found: {root}")
        
        dirs = get_data_directories()
        print("\nData directories:")
        for name, path in dirs.items():
            exists = "✅" if os.path.exists(path) else "❌"
            print(f"  {name:20}: {exists} {path}")
            
    except RuntimeError as e:
        print(f"Error: {e}")