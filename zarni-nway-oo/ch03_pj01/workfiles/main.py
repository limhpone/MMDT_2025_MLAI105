from classify_img import get_predictions
from analysis_visualizer import analyze_predictions_and_save_results
from fuzzy_evaluator import run_fuzzy_evaluation
import os

def run_prediction(name: str, image_dir: str, output_csv: str):
    """
    Run predictions on images in a directory and save results to CSV
    
    Args:
        name: Name of the dataset for logging
        image_dir: Directory containing images to classify
        output_csv: Path to save the predictions CSV file
    """
    if not os.path.exists(image_dir):
        raise ValueError(f"Image directory '{image_dir}' does not exist.")
    if not os.path.exists(os.path.dirname(output_csv)):
        os.makedirs(os.path.dirname(output_csv))
    
    print(f"Starting predictions for '{name}' from {image_dir}")
    results_df = get_predictions(image_dir, 3)
    results_df.to_csv(output_csv, index=False)
    print(f"Saved predictions for '{name}' to '{output_csv}'")
    return results_df

def run_prediction_with_analysis(name: str, image_dir: str, dataset_folder_name: str):
    """
    Run predictions and automatically generate analysis plots and reports
    
    Args:
        name: Name of the dataset for logging
        image_dir: Directory containing images to classify
        dataset_folder_name: Name used for organizing results
    """
    # Define paths
    output_csv = f"./results/{dataset_folder_name}/{dataset_folder_name}_predictions.csv"
    
    try:
        # Run predictions
        print("=" * 60)
        print(f"STEP 1: Running CNN Predictions for {name}")
        print("=" * 60)
        results_df = run_prediction(name, image_dir, output_csv)
        print(f"✓ Predictions completed: {len(results_df)} images processed")
        
        # Run analysis and generate visualizations
        print("\n" + "=" * 60)
        print(f"STEP 2: Analyzing Results and Generating Visualizations")
        print("=" * 60)
        results_dir, plot_files, report_path = analyze_predictions_and_save_results(
            csv_file_path=output_csv,
            dataset_folder_name=dataset_folder_name,
            results_base_dir="./results"
        )
        
        print(f"✓ Analysis completed successfully!")
        print(f"✓ Results directory: {results_dir}")
        print(f"✓ Generated {len(plot_files)} visualization plots")
        print(f"✓ Generated analysis report: {report_path}")
        
        # Run fuzzy evaluation
        print("\n" + "=" * 60)
        print(f"STEP 3: Running Fuzzy Similarity Evaluation")
        print("=" * 60)
        fuzzy_results, fuzzy_report_path = run_fuzzy_evaluation(
            csv_file_path=output_csv,
            dataset_name=dataset_folder_name,
            results_base_dir="./results"
        )
        
        print(f"✓ Fuzzy evaluation completed successfully!")
        print(f"✓ Fuzzy evaluation report: {fuzzy_report_path}")
        
        return results_df, results_dir, plot_files, report_path, fuzzy_results, fuzzy_report_path
        
    except Exception as e:
        print(f"❌ Error during prediction and analysis: {str(e)}")
        raise

def run_multiple_datasets():
    """
    Function to run predictions and analysis on multiple datasets
    Modify this function to add more datasets as needed
    """
    datasets = [
        {
            "name": "Mammals",
            "image_dir": "./dataset/mammals/",
            "folder_name": "mammals"
        },
        # Add more datasets here as needed:
        {
            "name": "Blurry_Noisy",
            "image_dir": "./dataset/blurry_noisy/",
            "folder_name": "blurry_noisy"
        },
        {
            "name": "MMCulture",
            "image_dir": "./dataset/mmculture/",
            "folder_name": "mmculture"
        },
    ]
    
    all_results = {}
    
    for dataset in datasets:
        try:
            print(f"\n{'='*80}")
            print(f"PROCESSING DATASET: {dataset['name'].upper()}")
            print(f"{'='*80}")
            
            results = run_prediction_with_analysis(
                name=dataset["name"],
                image_dir=dataset["image_dir"],
                dataset_folder_name=dataset["folder_name"]
            )
            
            all_results[dataset["name"]] = {
                "status": "SUCCESS",
                "results": results
            }
            
        except Exception as e:
            print(f"❌ Failed to process {dataset['name']}: {str(e)}")
            all_results[dataset["name"]] = {
                "status": "FAILED",
                "error": str(e)
            }
            continue  # Continue with next dataset
    
    # Print summary
    print(f"\n{'='*80}")
    print("PROCESSING SUMMARY")
    print(f"{'='*80}")
    
    for dataset_name, result in all_results.items():
        status_icon = "✅" if result["status"] == "SUCCESS" else "❌"
        print(f"{status_icon} {dataset_name}: {result['status']}")
        if result["status"] == "FAILED":
            print(f"    Error: {result['error']}")
    
    return all_results

def analyze_existing_csv(csv_file_path: str, dataset_folder_name: str):
    """
    Analyze existing CSV file and generate visualizations with fuzzy evaluation
    
    Args:
        csv_file_path: Path to existing CSV file with predictions
        dataset_folder_name: Name for organizing results (e.g., "mammals", "birds")
    """
    print("=" * 60)
    print(f"ANALYZING EXISTING CSV: {csv_file_path}")
    print("=" * 60)
    
    try:
        # Check if file exists
        if not os.path.exists(csv_file_path):
            raise FileNotFoundError(f"CSV file not found: {csv_file_path}")
        
        # Step 1: Run standard analysis
        print("🔍 STEP 1: Running Standard Analysis and Visualizations")
        print("-" * 40)
        results_dir, plot_files, report_path = analyze_predictions_and_save_results(
            csv_file_path=csv_file_path,
            dataset_folder_name=dataset_folder_name,
            results_base_dir="./results"
        )
        
        print(f"✅ Standard analysis completed!")
        print(f"📁 Results directory: {results_dir}")
        print(f"📊 Generated {len(plot_files)} visualization plots:")
        for plot_file in plot_files:
            print(f"   - {os.path.basename(plot_file)}")
        print(f"📋 Analysis report: {report_path}")
        
        # Step 2: Run fuzzy evaluation
        print(f"\n🔍 STEP 2: Running Fuzzy Similarity Evaluation")
        print("-" * 40)
        fuzzy_results, fuzzy_report_path = run_fuzzy_evaluation(
            csv_file_path=csv_file_path,
            dataset_name=dataset_folder_name,
            results_base_dir="./results"
        )
        
        print(f"✅ Fuzzy evaluation completed!")
        print(f"📋 Fuzzy evaluation report: {fuzzy_report_path}")
        
        return results_dir, plot_files, report_path, fuzzy_results, fuzzy_report_path
        
    except Exception as e:
        print(f"❌ Error analyzing CSV: {str(e)}")
        raise

def main():
    """Main function to run predictions and analysis"""
    try:
        # dataset
        results = run_prediction_with_analysis(
            name="Mammals", 
            image_dir="./dataset/mammals/", 
            dataset_folder_name="mammals"
        )
        
        print("\n" + "=" * 60)
        print("🎉 PROCESSING COMPLETED SUCCESSFULLY!")
        
    except Exception as e:
        print(f"\n❌ PROCESSING FAILED: {str(e)}")

if __name__ == "__main__":
    # Run single dataset
    # main()
    
    # Run multiple datasets
    # run_multiple_datasets()
    
    #Analyze existing .csv files
    # analyze_existing_csv("./results/mammals/mammals_predictions.csv", "mammals")
    # analyze_existing_csv("./results/mmculture/mmculture_predictions.csv", "mmculture")
    analyze_existing_csv("./results/blurry_noisy/blurry_noisy_predictions.csv", "blurry_noisy")
    