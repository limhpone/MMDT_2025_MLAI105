import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple, Union
import time

# Set up professional styling
plt.style.use('default')
sns.set_palette("husl")

class CNNAnalyzer:
    def __init__(self, predictions_df: pd.DataFrame):
        """
        Initialize CNN analyzer with predictions DataFrame
        Auto-detects top_k value from available columns
        """
        self.predictions_df = predictions_df
        self.models = ['ResNet50', 'VGGNet16', 'InceptionV3', 'ConvNeXt', 'EfficientNet']
        self.inception_color = '#FF6B35'
        self.model_colors = {
            'ResNet50': '#4ECDC4',
            'VGGNet16': '#45B7D1', 
            'InceptionV3': '#FF6B35',
            'ConvNeXt': '#96CEB4',
            'EfficientNet': '#FECA57'
        }
        
        # Auto-detect top_k value from available columns
        self.top_k = self._detect_top_k()
        print(f"📊 Auto-detected top_k value: {self.top_k}")
        
    def _detect_top_k(self) -> int:
        """
        Auto-detect the top_k value from available columns
        """
        # Check for top1, top2, top3... columns for the first model
        first_model = self.models[0]
        k_values = []
        
        for k in range(1, 6):  # Check up to top5
            col_name = f"{first_model}_top{k}_prob"
            if col_name in self.predictions_df.columns:
                k_values.append(k)
        
        if not k_values:
            raise ValueError("No top-k probability columns found in the DataFrame")
        
        return max(k_values)
        
    def extract_model_data(self) -> Dict:
        """Extract and organize prediction data by model (flexible for any top_k)"""
        model_data = {}
        
        for model in self.models:
            # Always include top1
            top1_col = f"{model}_top1"
            top1_prob_col = f"{model}_top1_prob"
            
            if top1_col not in self.predictions_df.columns or top1_prob_col not in self.predictions_df.columns:
                print(f"⚠️  Warning: Missing columns for model {model}, skipping...")
                continue
            
            data = {
                'top1_predictions': self.predictions_df[top1_col].tolist(),
                'top1_probs': self.predictions_df[top1_prob_col].tolist(),
            }
            
            # Add additional top-k columns if they exist
            all_probs = self.predictions_df[top1_prob_col].tolist()
            
            for k in range(2, self.top_k + 1):
                topk_prob_col = f"{model}_top{k}_prob"
                if topk_prob_col in self.predictions_df.columns:
                    data[f'top{k}_probs'] = self.predictions_df[topk_prob_col].tolist()
                    all_probs.extend(self.predictions_df[topk_prob_col].tolist())
            
            data['all_probs'] = all_probs
            model_data[model] = data
            
        return model_data
    
    def calculate_performance_metrics(self) -> pd.DataFrame:
        """Calculate performance metrics for each model"""
        model_data = self.extract_model_data()
        metrics = []
        
        for model in self.models:
            if model not in model_data:
                # Add default empty metrics for missing models
                metrics.append({
                    'model': model,
                    'avg_top1_confidence': 0.0,
                    'avg_all_confidence': 0.0,
                    'std_top1_confidence': 0.0,
                    'high_confidence_pct': 0.0,
                    'low_confidence_pct': 0.0,
                    'total_predictions': 0
                })
                continue
                
            data = model_data[model]
            
            # Calculate metrics
            avg_top1_confidence = np.mean(data['top1_probs'])
            avg_all_confidence = np.mean(data['all_probs'])
            std_top1_confidence = np.std(data['top1_probs'])
            
            # High confidence predictions (>0.8)
            high_conf_count = sum(1 for p in data['top1_probs'] if p > 0.8)
            high_conf_pct = (high_conf_count / len(data['top1_probs'])) * 100
            
            # Low confidence predictions (<0.3)
            low_conf_count = sum(1 for p in data['top1_probs'] if p < 0.3)
            low_conf_pct = (low_conf_count / len(data['top1_probs'])) * 100
            
            metrics.append({
                'model': model,
                'avg_top1_confidence': avg_top1_confidence,
                'avg_all_confidence': avg_all_confidence,
                'std_top1_confidence': std_top1_confidence,
                'high_confidence_pct': high_conf_pct,
                'low_confidence_pct': low_conf_pct,
                'total_predictions': len(data['top1_probs'])
            })
        
        return pd.DataFrame(metrics)
    
    def calculate_timing_metrics(self) -> pd.DataFrame:
        """Calculate timing metrics from the predictions DataFrame"""
        timing_metrics = []
        
        for model in self.models:
            time_col = f"{model}_time"
            
            if time_col in self.predictions_df.columns:
                # Use actual timing data
                times = self.predictions_df[time_col]
                avg_time = times.mean()
                std_time = times.std()
                min_time = times.min()
                max_time = times.max()
            else:
                # Set default values for missing timing data
                print(f"⚠️  Warning: No timing data found for {model}, using default values")
                avg_time = 1.0  # Default 1 second
                std_time = 0.1
                min_time = 0.9
                max_time = 1.1
            
            timing_metrics.append({
                'model': model,
                'avg_time': avg_time,
                'std_time': std_time,
                'min_time': min_time,
                'max_time': max_time
            })
        
        return pd.DataFrame(timing_metrics)
    
    def calculate_confidence_metrics(self) -> pd.DataFrame:
        """Calculate detailed confidence metrics"""
        model_data = self.extract_model_data()
        confidence_metrics = []
        
        for model in self.models:
            if model not in model_data:
                # Add default confidence metrics for missing models
                confidence_metrics.append({
                    'model': model,
                    'mean_confidence': 0.0,
                    'median_confidence': 0.0,
                    'q25_confidence': 0.0,
                    'q75_confidence': 0.0,
                    'min_confidence': 0.0,
                    'max_confidence': 0.0
                })
                continue
                
            data = model_data[model]
            top1_probs = data['top1_probs']
            
            confidence_metrics.append({
                'model': model,
                'mean_confidence': np.mean(top1_probs),
                'median_confidence': np.median(top1_probs),
                'q25_confidence': np.percentile(top1_probs, 25),
                'q75_confidence': np.percentile(top1_probs, 75),
                'min_confidence': np.min(top1_probs),
                'max_confidence': np.max(top1_probs)
            })
        
        return pd.DataFrame(confidence_metrics)
    
    def create_summary_dataframe(self) -> pd.DataFrame:
        """Create comprehensive summary DataFrame"""
        perf_metrics = self.calculate_performance_metrics()
        timing_metrics = self.calculate_timing_metrics()
        conf_metrics = self.calculate_confidence_metrics()
        
        # Merge all metrics using left join to preserve all models
        summary = perf_metrics.merge(timing_metrics, on='model', how='left')
        summary = summary.merge(conf_metrics, on='model', how='left')
        
        # Fill any missing values with reasonable defaults
        summary['avg_time'] = summary['avg_time'].fillna(1.0)  # Default 1 second
        summary = summary.fillna(0)
        
        # Calculate overall score (normalized) - protect against division by zero
        summary['speed_score'] = 1 / summary['avg_time'].replace(0, 1.0)  # Higher is better
        summary['confidence_score'] = summary['avg_top1_confidence']
        
        # Normalize scores to 0-100 (handle division by zero)
        for col in ['speed_score', 'confidence_score']:
            col_min = summary[col].min()
            col_max = summary[col].max()
            if col_max - col_min > 0:  # Avoid division by zero
                summary[f'{col}_normalized'] = ((summary[col] - col_min) / 
                                              (col_max - col_min)) * 100
            else:
                summary[f'{col}_normalized'] = 50  # Default middle value if all same
        
        # Calculate overall rank
        summary['overall_score'] = (summary['speed_score_normalized'] + 
                                  summary['confidence_score_normalized']) / 2
        summary['rank'] = summary['overall_score'].rank(method='dense', ascending=False).astype(int)
        
        return summary
    
    def plot_performance_overview(self, save_path: str) -> str:
        """Create multi-metric performance overview"""
        summary_df = self.create_summary_dataframe()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(summary_df))
        width = 0.25
        
        # Convert pandas Series to numpy arrays to fix type warnings
        confidence_norm = np.array(summary_df['confidence_score_normalized'].values, dtype=float)
        speed_norm = np.array(summary_df['speed_score_normalized'].values, dtype=float)
        overall_norm = np.array(summary_df['overall_score'].values, dtype=float)
        
        # Define colors for each metric type (professional and distinct)
        metric_colors = {
            'confidence': '#3498db',  # Professional blue
            'speed': '#e74c3c',       # Professional red
            'overall': '#27ae60'      # Professional green
        }
        
        # Create bars with distinct colors for each metric
        bars1 = ax.bar(x - width, confidence_norm, width, 
                       label='Confidence Score', 
                       color=metric_colors['confidence'], 
                       alpha=0.8, 
                       edgecolor='white', 
                       linewidth=1)
        
        bars2 = ax.bar(x, speed_norm, width, 
                       label='Speed Score', 
                       color=metric_colors['speed'], 
                       alpha=0.8, 
                       edgecolor='white', 
                       linewidth=1)
        
        bars3 = ax.bar(x + width, overall_norm, width, 
                       label='Overall Score', 
                       color=metric_colors['overall'], 
                       alpha=0.8, 
                       edgecolor='white', 
                       linewidth=1)
        
        # Highlight InceptionV3 with black border (optional)
        for i, model in enumerate(summary_df['model']):
            if model == 'InceptionV3':
                bars1[i].set_edgecolor('black')
                bars2[i].set_edgecolor('black')
                bars3[i].set_edgecolor('black')
                bars1[i].set_linewidth(2)
                bars2[i].set_linewidth(2)
                bars3[i].set_linewidth(2)
        
        # Add value labels on top of bars (optional - makes it more informative)
        for i in range(len(summary_df)):
            # Confidence score labels
            ax.text(x[i] - width, confidence_norm[i] + 1, f'{confidence_norm[i]:.1f}', 
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
            # Speed score labels
            ax.text(x[i], speed_norm[i] + 1, f'{speed_norm[i]:.1f}', 
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
            # Overall score labels
            ax.text(x[i] + width, overall_norm[i] + 1, f'{overall_norm[i]:.1f}', 
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('CNN Models', fontsize=12, fontweight='bold')
        ax.set_ylabel('Normalized Performance Score (0-100)', fontsize=12, fontweight='bold')
        ax.set_title('CNN Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(summary_df['model'], rotation=45)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')  # Only horizontal grid lines
        
        # Set y-axis limit to accommodate labels
        ax.set_ylim(0, max(max(confidence_norm), max(speed_norm), max(overall_norm)) + 10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_inference_time_comparison(self, save_path: str) -> str:
        """Create inference time comparison plot"""
        timing_df = self.calculate_timing_metrics()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sort by time
        timing_df = timing_df.sort_values('avg_time')
        
        # Create color array
        colors = []
        for model in timing_df['model']:
            colors.append(self.model_colors[model])
        
        # Convert to numpy arrays to fix type warnings
        avg_times = np.array(timing_df['avg_time'].values, dtype=float)
        std_times = np.array(timing_df['std_time'].values, dtype=float)
        
        bars = ax.barh(timing_df['model'], avg_times, 
                      color=colors, alpha=0.8)
        
        # Highlight InceptionV3
        for i, model in enumerate(timing_df['model']):
            if model == 'InceptionV3':
                bars[i].set_edgecolor('black')
                bars[i].set_linewidth(2)
        
        ax.set_xlabel('Inference Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_title('Model Inference Time Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (model, time) in enumerate(zip(timing_df['model'], avg_times)):
            ax.text(time + 0.005, i, f'{time:.3f}s', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_confidence_distribution(self, save_path: str) -> str:
        """Create confidence distribution box plot"""
        model_data = self.extract_model_data()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data - convert to numpy arrays to fix type warnings
        confidence_data = []
        model_names = []
        
        for model in self.models:
            if model in model_data:
                all_probs = np.array(model_data[model]['all_probs'], dtype=float)
                confidence_data.append(all_probs)
                model_names.append(model)
        
        # Create box plot
        bp = ax.boxplot(confidence_data, patch_artist=True,
                        medianprops={'color': 'black', 'linewidth': 1.5})
        
        # Set x-axis labels manually
        ax.set_xticklabels(model_names, rotation=45)
        
        # Color the boxes
        for i, patch in enumerate(bp['boxes']):
            model = model_names[i]
            patch.set_facecolor(self.model_colors[model])
            patch.set_alpha(0.7)
            
            if model == 'InceptionV3':
                patch.set_edgecolor('black')
                patch.set_linewidth(2)
        
        ax.set_xlabel('CNN Models', fontsize=12, fontweight='bold')
        ax.set_ylabel('Prediction Confidence (Probability)', fontsize=12, fontweight='bold')
        ax.set_title(f'Top-{self.top_k} Prediction Confidence Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_inception_analysis(self, save_path: str) -> str:
        """Create detailed InceptionV3 analysis"""
        model_data = self.extract_model_data()
        
        if 'InceptionV3' not in model_data:
            print("⚠️  InceptionV3 data not found, skipping inception analysis")
            return save_path
            
        inception_data = model_data['InceptionV3']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('InceptionV3 Detailed Analysis', fontsize=16, fontweight='bold')
        
        # Convert to numpy arrays to fix type warnings
        top1_probs = np.array(inception_data['top1_probs'], dtype=float)
        
        # Panel 1: Top-1 confidence histogram
        ax1.hist(top1_probs, bins=20, color=self.inception_color, 
                alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Top-1 Confidence Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('InceptionV3 Top-1 Confidence Distribution')
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Top-K confidence comparison
        k_values = []
        avg_confidences = []
        
        for k in range(1, self.top_k + 1):
            if f'top{k}_probs' in inception_data:
                topk_probs = np.array(inception_data[f'top{k}_probs'], dtype=float)
                k_values.append(k)
                avg_confidences.append(np.mean(topk_probs))
        
        if k_values:
            ax2.plot(k_values, avg_confidences, 'o-', color=self.inception_color, 
                    linewidth=3, markersize=8)
            ax2.set_xlabel('Top-K Rank')
            ax2.set_ylabel('Average Confidence')
            ax2.set_title('InceptionV3 Average Confidence by Rank')
            ax2.grid(True, alpha=0.3)
            ax2.set_xticks(k_values)
        else:
            ax2.text(0.5, 0.5, 'No Top-K data available', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=12)
            ax2.set_title('InceptionV3 Top-K Analysis (No Data)')
        
        # Panel 3: Confidence by prediction rank (flexible for any top_k)
        rank_data = [top1_probs]
        rank_labels = ['Top-1']
        
        for k in range(2, self.top_k + 1):
            if f'top{k}_probs' in inception_data:
                topk_probs = np.array(inception_data[f'top{k}_probs'], dtype=float)
                rank_data.append(topk_probs)
                rank_labels.append(f'Top-{k}')
        
        ax3.boxplot(rank_data, patch_artist=True, labels=rank_labels,
                    medianprops={'color': 'black', 'linewidth': 1.5})
        ax3.set_ylabel('Confidence Score')
        ax3.set_title('InceptionV3 Confidence by Prediction Rank')
        ax3.grid(True, alpha=0.3)
        
        # Panel 4: Confidence categories
        high_conf = sum(1 for p in top1_probs if p > 0.8)
        med_conf = sum(1 for p in top1_probs if 0.3 <= p <= 0.8)
        low_conf = sum(1 for p in top1_probs if p < 0.3)
        
        categories = ['High\n(>0.8)', 'Medium\n(0.3-0.8)', 'Low\n(<0.3)']
        counts = [high_conf, med_conf, low_conf]
        colors = ['#2E8B57', '#FFD700', '#DC143C']
        
        ax4.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
        ax4.set_ylabel('Number of Predictions')
        ax4.set_title('InceptionV3 Confidence Categories')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def create_summary_table_plot(self, save_path: str) -> str:
        """Create professional summary table"""
        summary_df = self.create_summary_dataframe()
        
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare table data
        table_data = []
        for _, row in summary_df.iterrows():
            table_row = [
                row['model'],
                f"{row['avg_top1_confidence']:.3f}",
                f"{row['avg_time']:.3f}s",
                f"{row['high_confidence_pct']:.1f}%",
                f"{row['std_top1_confidence']:.3f}",
                f"#{row['rank']}"
            ]
            table_data.append(table_row)
        
        # Create table
        table = ax.table(cellText=table_data,
                        colLabels=['Model', 'Avg Confidence', 'Avg Time', 'High Conf %', 'Std Dev', 'Rank'],
                        cellLoc='center',
                        loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)
        
        # Style table
        for i in range(len(table_data)):
            model = table_data[i][0]
            for j in range(len(table_data[i])):
                cell = table[(i + 1, j)]
                cell.set_facecolor(self.model_colors[model])
                cell.set_alpha(0.3)
                
                if model == 'InceptionV3':
                    cell.set_facecolor(self.inception_color)
                    cell.set_alpha(0.7)
                    cell.set_text_props(weight='bold')
        
        # Style header
        for j in range(len(table_data[0])):
            table[(0, j)].set_facecolor('#4472C4')
            table[(0, j)].set_text_props(weight='bold', color='white')
        
        plt.title('CNN Model Performance Summary', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def generate_all_plots(self, plots_dir: str, dataset_name: str) -> List[str]:
        """Generate all visualization plots"""
        # Create directory if it doesn't exist
        os.makedirs(plots_dir, exist_ok=True)
        
        plot_files = []
        
        plot1 = os.path.join(plots_dir, f'{dataset_name}_performance_overview.png')
        self.plot_performance_overview(plot1)
        plot_files.append(plot1)
        
        plot2 = os.path.join(plots_dir, f'{dataset_name}_inference_time.png')
        self.plot_inference_time_comparison(plot2)
        plot_files.append(plot2)
        
        plot3 = os.path.join(plots_dir, f'{dataset_name}_confidence_distribution.png')
        self.plot_confidence_distribution(plot3)
        plot_files.append(plot3)
        
        plot4 = os.path.join(plots_dir, f'{dataset_name}_inception_analysis.png')
        self.plot_inception_analysis(plot4)
        plot_files.append(plot4)
        
        plot5 = os.path.join(plots_dir, f'{dataset_name}_summary_table.png')
        self.create_summary_table_plot(plot5)
        plot_files.append(plot5)
        
        return plot_files
    
    def generate_analysis_report(self, report_path: str, dataset_name: str):
        """Generate markdown analysis report"""
        summary_df = self.create_summary_dataframe()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write(f"# CNN Model Analysis Report\n\n")
            f.write(f"**Dataset:** {dataset_name.upper()}  \n")
            f.write(f"**Top-K Value:** {self.top_k}  \n")
            f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}  \n\n")
            f.write(f"---\n\n")
            
            f.write("## Performance Summary\n\n")
            
            # Create a performance summary table
            f.write("| Model | Rank | Avg Confidence | Avg Time (s) | High Confidence % | Overall Score |\n")
            f.write("|-------|------|----------------|--------------|-------------------|---------------|\n")
            
            for _, row in summary_df.iterrows():
                f.write(f"| {row['model']} | #{row['rank']} | {row['avg_top1_confidence']:.3f} | {row['avg_time']:.3f} | {row['high_confidence_pct']:.1f}% | {row['overall_score']:.1f} |\n")
            
            f.write("\n")
            
            # InceptionV3 specific analysis
            inception_rows = summary_df[summary_df['model'] == 'InceptionV3']
            if not inception_rows.empty:
                inception_row = inception_rows.iloc[0]
                f.write("## InceptionV3 Analysis\n\n")
                f.write(f"- **Rank:** #{inception_row['rank']} out of {len(summary_df)} models\n")
                f.write(f"- **Confidence Score:** {inception_row['avg_top1_confidence']:.3f}\n")
                f.write(f"- **Speed Performance:** {inception_row['avg_time']:.3f}s average\n")
                f.write(f"- **Reliability:** {inception_row['high_confidence_pct']:.1f}% high confidence predictions\n\n")
            
            f.write("## Recommendations\n\n")
            best_model = summary_df.loc[summary_df['rank'] == 1, 'model'].iloc[0]
            fastest_model = summary_df.loc[summary_df['avg_time'].idxmin(), 'model']
            most_confident = summary_df.loc[summary_df['avg_top1_confidence'].idxmax(), 'model']
            
            f.write(f"- **Best Overall:** {best_model}\n")
            f.write(f"- **Fastest:** {fastest_model}\n")
            f.write(f"- **Most Confident:** {most_confident}\n")


def analyze_predictions_and_save_results(csv_file_path: str, dataset_folder_name: str, results_base_dir: str = "./results"):
    """
    Load predictions from CSV, analyze them, and save plots and reports to results directory
    
    Args:
        csv_file_path: Path to the CSV file containing predictions
        dataset_folder_name: Name of the dataset folder (used for naming outputs)
        results_base_dir: Base directory for results (default: "./results")
    """
    try:
        # Load predictions DataFrame
        print(f"Loading predictions from: {csv_file_path}")
        predictions_df = pd.read_csv(csv_file_path)
        print(f"Loaded {len(predictions_df)} predictions")
        
        # Create analyzer (will auto-detect top_k)
        analyzer = CNNAnalyzer(predictions_df)
        
        # Create results directory structure
        results_dir = os.path.join(results_base_dir, dataset_folder_name)
        plots_dir = os.path.join(results_dir, "plots")
        reports_dir = os.path.join(results_dir, "reports")
        
        # Generate all plots
        print(f"Generating visualization plots...")
        plot_files = analyzer.generate_all_plots(plots_dir, dataset_folder_name)
        print(f"Generated {len(plot_files)} plot files:")
        for plot_file in plot_files:
            print(f"  - {plot_file}")
        
        # Generate analysis report
        report_path = os.path.join(reports_dir, f"{dataset_folder_name}_analysis_report.md")
        print(f"Generating analysis report...")
        analyzer.generate_analysis_report(report_path, dataset_folder_name)
        print(f"Generated analysis report: {report_path}")
        
        print(f"\nAnalysis complete! Results saved to: {results_dir}")
        return results_dir, plot_files, report_path
        
    except FileNotFoundError:
        print(f"Error: Could not find predictions file: {csv_file_path}")
        raise
    except Exception as e:
        print(f"Error analyzing predictions: {str(e)}")
        raise