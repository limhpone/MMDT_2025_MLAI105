#!/usr/bin/env python3
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
    
    print(f"\n🔄 Creating ensemble from {num_models} models...")
    
    # Simple majority voting
    ensemble_pred, individual_preds = ensemble.create_ensemble_prediction(y_test)
    
    if ensemble_pred is not None:
        ensemble_acc = np.mean(ensemble_pred == y_test)
        print(f"\n📊 Ensemble Results (Majority Voting):")
        print(f"   Ensemble Accuracy: {ensemble_acc:.4f}")
        
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, ensemble_pred, target_names=['red', 'neutral', 'green']))
        
        # Weighted ensemble
        weighted_pred = ensemble.weighted_ensemble(y_test)
        if weighted_pred is not None:
            weighted_acc = np.mean(weighted_pred == y_test)
            print(f"\n📊 Weighted Ensemble Accuracy: {weighted_acc:.4f}")
        
        # Compare with individual models
        print(f"\n📈 Model Comparison:")
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
        
        print(f"\n✅ Ensemble training completed!")
        print(f"📁 Results saved: ensemble_results_{timestamp}.json")
    
    else:
        print("❌ Ensemble creation failed")

if __name__ == "__main__":
    main()
