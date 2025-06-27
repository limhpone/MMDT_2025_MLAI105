import pandas as pd
import numpy as np
import re
from difflib import SequenceMatcher
from typing import Dict, List, Tuple
import os

# Try to import thefuzz (updated version of fuzzywuzzy), fall back to difflib if not available
try:
    from thefuzz import fuzz
    THEFUZZ_AVAILABLE = True
except ImportError:
    THEFUZZ_AVAILABLE = False
    print("⚠️  thefuzz not available, using difflib for fuzzy matching")

class AutoSimilarityEvaluator:
    """Universal similarity evaluator for any dataset"""
    
    def __init__(self, threshold_high=0.6, threshold_medium=0.3):
        self.threshold_high = threshold_high
        self.threshold_medium = threshold_medium
        self.models = ['ResNet50', 'VGGNet16', 'InceptionV3', 'ConvNeXt', 'EfficientNet']
    
    def clean_label(self, label):
        """Standardize labels for comparison"""
        if pd.isna(label):
            return ""
        return re.sub(r'[_-]', ' ', str(label).lower().strip())
    
    def calculate_word_overlap(self, text1, text2):
        """Calculate word-level overlap using Jaccard similarity"""
        text1_clean = self.clean_label(text1)
        text2_clean = self.clean_label(text2)
        
        words1 = set(text1_clean.split())
        words2 = set(text2_clean.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def ngram_similarity(self, text1, text2, n=2):
        """Character n-gram similarity for catching morphological relationships"""
        text1_clean = self.clean_label(text1)
        text2_clean = self.clean_label(text2)
        
        if len(text1_clean) < n or len(text2_clean) < n:
            return 0.0
        
        ngrams1 = set([text1_clean[i:i+n] for i in range(len(text1_clean)-n+1)])
        ngrams2 = set([text2_clean[i:i+n] for i in range(len(text2_clean)-n+1)])
        
        if not ngrams1 or not ngrams2:
            return 0.0
        
        intersection = ngrams1.intersection(ngrams2)
        union = ngrams1.union(ngrams2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def analyze_compound_similarity(self, text1, text2):
        """Handle compound labels like 'brown_bear' vs 'ice_bear'"""
        parts1 = re.split(r'[_\s-]', self.clean_label(text1))
        parts2 = re.split(r'[_\s-]', self.clean_label(text2))
        
        # Filter out very short words (likely articles/prepositions)
        significant1 = [p for p in parts1 if len(p) > 2]
        significant2 = [p for p in parts2 if len(p) > 2]
        
        if not significant1 or not significant2:
            return 0.0
        
        shared = set(significant1).intersection(set(significant2))
        total_unique = set(significant1).union(set(significant2))
        
        return len(shared) / len(total_unique) if total_unique else 0.0
    
    def fuzzy_ratio(self, text1, text2):
        """Calculate fuzzy string similarity ratio"""
        if THEFUZZ_AVAILABLE:
            return fuzz.ratio(str(text1), str(text2)) / 100.0
        else:
            # Fallback to difflib
            return SequenceMatcher(None, str(text1), str(text2)).ratio()
    
    def evaluate_prediction(self, prediction, ground_truth):
        """Multi-method similarity scoring"""
        
        if pd.isna(prediction) or pd.isna(ground_truth):
            return 0.0, {}
        
        # Exact match check first
        if str(prediction).lower() == str(ground_truth).lower():
            return 1.0, {'exact_match': True}
        
        scores = {}
        
        # Method 1: Fuzzy string similarity
        scores['fuzzy'] = self.fuzzy_ratio(prediction, ground_truth)
        
        # Method 2: Word overlap
        scores['word_overlap'] = self.calculate_word_overlap(prediction, ground_truth)
        
        # Method 3: N-gram similarity
        scores['ngram'] = self.ngram_similarity(prediction, ground_truth)
        
        # Method 4: Compound analysis
        scores['compound'] = self.analyze_compound_similarity(prediction, ground_truth)
        
        # Weighted combination (you can adjust these weights)
        final_score = (
            scores['fuzzy'] * 0.4 +
            scores['word_overlap'] * 0.3 +
            scores['ngram'] * 0.2 +
            scores['compound'] * 0.1
        )
        
        scores['final_score'] = final_score
        scores['exact_match'] = False
        
        return final_score, scores
    
    def categorize_similarity(self, score):
        """Convert similarity score to category"""
        if score >= self.threshold_high:
            return 'high'      # Very similar (0.8+)
        elif score >= self.threshold_medium:
            return 'medium'    # Somewhat similar (0.5-0.8)
        else:
            return 'low'       # Not similar (<0.5)
    
    def calculate_fuzzy_topk_accuracy(self, predictions_df):
        """Calculate Top-K accuracy with fuzzy matching"""
        
        results = {}
        total_samples = len(predictions_df)
        
        if total_samples == 0:
            print("⚠️  No data to evaluate")
            return results
        
        for model in self.models:
            # Check if model columns exist
            top1_col = f'{model}_top1'
            if top1_col not in predictions_df.columns:
                print(f"⚠️  Columns for {model} not found, skipping...")
                continue
            
            model_results = {
                'exact_accuracy': {'top1': 0, 'top2': 0, 'top3': 0},
                'fuzzy_accuracy': {'top1': 0, 'top2': 0, 'top3': 0},
                'weighted_fuzzy_accuracy': {'top1': 0, 'top2': 0, 'top3': 0},
                'similarity_breakdown': {'high': 0, 'medium': 0, 'low': 0, 'exact': 0}
            }
            
            for idx, row in predictions_df.iterrows():
                ground_truth = row['label']
                
                # Track if we found any match for similarity breakdown
                found_any_match = False
                best_similarity_category = 'low'
                
                # Check Top-K predictions - accumulate for each level
                top_k_predictions = []
                for k in range(1, 4):  # Top-1, Top-2, Top-3
                    pred_col = f'{model}_top{k}'
                    if pred_col not in predictions_df.columns:
                        break
                    
                    prediction = row[pred_col]
                    top_k_predictions.append(prediction)
                    
                    # Check if ground truth is in top-k predictions so far
                    if ground_truth in top_k_predictions:
                        model_results['exact_accuracy'][f'top{k}'] += 1
                        model_results['fuzzy_accuracy'][f'top{k}'] += 1.0 #type: ignore
                        model_results['weighted_fuzzy_accuracy'][f'top{k}'] += 1.0 #type: ignore
                        found_any_match = True
                        best_similarity_category = 'exact'
                    else:
                        # Check fuzzy matches for all predictions in top-k
                        best_fuzzy_score = 0
                        for pred in top_k_predictions:
                            similarity_score, score_details = self.evaluate_prediction(pred, ground_truth)
                            if similarity_score > best_fuzzy_score:
                                best_fuzzy_score = similarity_score
                        
                        # If we have a good fuzzy match in top-k
                        if best_fuzzy_score > 0:
                            similarity_category = self.categorize_similarity(best_fuzzy_score)
                            
                            if similarity_category in ['high', 'medium']:
                                model_results['fuzzy_accuracy'][f'top{k}'] += 1.0 #type: ignore
                                model_results['weighted_fuzzy_accuracy'][f'top{k}'] += best_fuzzy_score #type: ignore
                                found_any_match = True
                                if similarity_category == 'high' or best_similarity_category == 'low':
                                    best_similarity_category = similarity_category
                
                # Record the best similarity found for this sample
                if found_any_match:
                    model_results['similarity_breakdown'][best_similarity_category] += 1
                else:
                    model_results['similarity_breakdown']['low'] += 1
            
            # Convert to percentages
            for metric in ['exact_accuracy', 'fuzzy_accuracy']:
                for k in ['top1', 'top2', 'top3']:
                    model_results[metric][k] = (model_results[metric][k] / total_samples) * 100 #type: ignore
            
            # Weighted fuzzy accuracy (average similarity score)
            for k in ['top1', 'top2', 'top3']:
                model_results['weighted_fuzzy_accuracy'][k] = ( #type: ignore
                    model_results['weighted_fuzzy_accuracy'][k] / total_samples
                ) * 100
            
            results[model] = model_results
        
        return results
    
    def generate_fuzzy_evaluation_report(self, results, dataset_name, save_path):
        """Generate detailed fuzzy evaluation report in markdown format"""
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            f.write(f"# Fuzzy Evaluation Report\n\n")
            f.write(f"**Dataset:** {dataset_name.upper()}  \n")
            f.write(f"**Evaluation Method:** Multi-Fuzzy Similarity Analysis  \n\n")
            f.write(f"---\n\n")
            
            f.write("## Evaluation Methodology\n\n")
            f.write("- **Exact Match:** Identical predictions (100% score)\n")
            f.write("- **High Similarity:** 60%+ fuzzy match (e.g., 'snow_leopard' vs 'leopard')\n")
            f.write("- **Medium Similarity:** 30-60% fuzzy match (e.g., 'ice_bear' vs 'polar_bear')\n")
            f.write("- **Low Similarity:** <30% fuzzy match (unrelated terms)\n\n")
            
            f.write("## Fuzzy Accuracy Results\n\n")
            
            # Create summary table
            f.write("| Model | Exact Top-1 | Fuzzy Top-1 | Fuzzy Top-3 | Weighted Score |\n")
            f.write("|-------|-------------|-------------|-------------|----------------|\n")
            
            for model, result in results.items():
                exact_top1 = result['exact_accuracy']['top1']
                fuzzy_top1 = result['fuzzy_accuracy']['top1']
                fuzzy_top3 = result['fuzzy_accuracy']['top3']
                weighted_top1 = result['weighted_fuzzy_accuracy']['top1']
                
                f.write(f"| {model} | {exact_top1:.1f}% | {fuzzy_top1:.1f}% | {fuzzy_top3:.1f}% | {weighted_top1:.1f}% |\n")
            
            f.write("\n---\n\n")
            
            # Detailed breakdown
            f.write("## Detailed Model Analysis\n\n")
            
            for model, result in results.items():
                f.write(f"### {model.upper()}\n\n")
                
                f.write("**Exact Accuracy:**\n")
                f.write(f"- Top-1: {result['exact_accuracy']['top1']:.1f}%\n")
                f.write(f"- Top-2: {result['exact_accuracy']['top2']:.1f}%\n")
                f.write(f"- Top-3: {result['exact_accuracy']['top3']:.1f}%\n\n")
                
                f.write("**Fuzzy Accuracy:**\n")
                f.write(f"- Top-1: {result['fuzzy_accuracy']['top1']:.1f}%\n")
                f.write(f"- Top-2: {result['fuzzy_accuracy']['top2']:.1f}%\n")
                f.write(f"- Top-3: {result['fuzzy_accuracy']['top3']:.1f}%\n\n")
                
                f.write("**Similarity Breakdown:**\n")
                f.write(f"- Exact: {result['similarity_breakdown']['exact']}\n")
                f.write(f"- High: {result['similarity_breakdown']['high']}\n")
                f.write(f"- Medium: {result['similarity_breakdown']['medium']}\n")
                f.write(f"- Low: {result['similarity_breakdown']['low']}\n\n")
            
            f.write("## Recommendations\n\n")
            
            # Find best performing models
            best_exact = max(results.items(), key=lambda x: x[1]['exact_accuracy']['top1'])
            best_fuzzy = max(results.items(), key=lambda x: x[1]['fuzzy_accuracy']['top1'])
            best_weighted = max(results.items(), key=lambda x: x[1]['weighted_fuzzy_accuracy']['top1'])
            
            f.write(f"- **Best Exact Accuracy:** {best_exact[0]} ({best_exact[1]['exact_accuracy']['top1']:.1f}%)\n")
            f.write(f"- **Best Fuzzy Accuracy:** {best_fuzzy[0]} ({best_fuzzy[1]['fuzzy_accuracy']['top1']:.1f}%)\n")
            f.write(f"- **Best Weighted Score:** {best_weighted[0]} ({best_weighted[1]['weighted_fuzzy_accuracy']['top1']:.1f}%)\n")


def run_fuzzy_evaluation(csv_file_path: str, dataset_name: str, results_base_dir: str = "./results"):
    """
    Run fuzzy evaluation on predictions CSV file
    
    Args:
        csv_file_path: Path to the CSV file containing predictions
        dataset_name: Name of the dataset folder
        results_base_dir: Base directory for results
    """
    try:
        print(f"🔍 Starting fuzzy evaluation for: {csv_file_path}")
        
        # Load predictions DataFrame
        predictions_df = pd.read_csv(csv_file_path)
        print(f"Loaded {len(predictions_df)} predictions for fuzzy analysis")
        
        # Create evaluator
        evaluator = AutoSimilarityEvaluator()
        
        # Calculate fuzzy accuracy
        fuzzy_results = evaluator.calculate_fuzzy_topk_accuracy(predictions_df)
        
        # Create results directory structure
        results_dir = os.path.join(results_base_dir, dataset_name)
        reports_dir = os.path.join(results_dir, "reports")
        
        # Generate fuzzy evaluation report
        report_path = os.path.join(reports_dir, f"{dataset_name}_fuzzy_evaluation_report.md")
        evaluator.generate_fuzzy_evaluation_report(fuzzy_results, dataset_name, report_path)
        
        print(f"📊 Fuzzy evaluation completed!")
        print(f"📄 Fuzzy evaluation report: {report_path}")
        
        # Print summary to console
        print(f"\n{'='*50}")
        print(f"FUZZY EVALUATION SUMMARY - {dataset_name.upper()}")
        print(f"{'='*50}")
        print(f"{'Model':<15} {'Exact Top-1':<12} {'Fuzzy Top-1':<12} {'Improvement':<12}")
        print("-" * 50)
        
        for model, result in fuzzy_results.items():
            exact_top1 = result['exact_accuracy']['top1']
            fuzzy_top1 = result['fuzzy_accuracy']['top1']
            improvement = fuzzy_top1 - exact_top1
            print(f"{model:<15} {exact_top1:<12.1f} {fuzzy_top1:<12.1f} +{improvement:<11.1f}")
        
        return fuzzy_results, report_path
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find predictions file: {csv_file_path}")
        raise
    except Exception as e:
        print(f"❌ Error in fuzzy evaluation: {str(e)}")
        raise