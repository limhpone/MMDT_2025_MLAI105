import os
import shutil
import pandas as pd

class MyanmarTextLabeller:
    def __init__(self):
        """Initialize Myanmar text labeller"""
        self.label_mapping = {
            'red': 0,
            'neutral': 1, 
            'green': 2
        }
    
    def determine_label_from_filename(self, filename):
        """Determine label based on filename"""
        if filename.startswith('red_'):
            return 'red', 0
        elif filename.startswith('green_'):
            return 'green', 2
        elif filename.startswith('neutral_'):
            return 'neutral', 1
        else:
            raise ValueError(f"Cannot determine label from filename: {filename}")
    
    def process_tokenized_file(self, input_file, output_file, category, label):
        """Process a tokenized file and add labels"""
        print(f"Labelling {category} articles from: {os.path.basename(input_file)}")
        
        # Read tokenized content
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        # Create labeled dataset
        labeled_data = []
        article_count = 0
        
        for line in lines:
            if line:  # Skip empty lines
                article_count += 1
                # Create short ID format
                short_label = category[0]  # r, g, n
                article_id = f"{short_label}_{article_count}"
                
                labeled_data.append({
                    'id': article_id,
                    'category': category,
                    'label': label,
                    'tokens': line,
                    'token_count': len(line.split())
                })
        
        # Create DataFrame and save as CSV
        df = pd.DataFrame(labeled_data)
        
        # Filter out entries with very few tokens
        df = df[df['token_count'] >= 5]  # Minimum 5 tokens
        
        # Save to CSV
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"Created {len(df)} labeled articles for {category}")
        print(f"Token count range: {df['token_count'].min()} - {df['token_count'].max()}")
        print(f"Average tokens per article: {df['token_count'].mean():.1f}")
        
        return len(df)

def main():
    """Process all tokenized files and create labeled datasets"""
    labeller = MyanmarTextLabeller()
    
    # Use relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, '..', '..')
    input_dir = os.path.join(project_root, "data", "tokenized", "to_process")
    output_dir = os.path.join(project_root, "data", "labelled", "to_process")
    done_dir = os.path.join(project_root, "data", "tokenized", "done")
    
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(done_dir, exist_ok=True)
    
    # Find all tokenized files
    import glob
    input_files = glob.glob(os.path.join(input_dir, "*.txt"))
    
    if not input_files:
        print(f"No tokenized files found in {input_dir}")
        return
    
    print(f"Processing {len(input_files)} tokenized files for labelling...")
    
    total_articles = 0
    all_datasets = []
    
    for input_file in input_files:
        try:
            # Determine category and label from filename
            filename = os.path.basename(input_file)
            category, label = labeller.determine_label_from_filename(filename)
            
            # Create output filename (CSV format)
            output_filename = filename.replace('.txt', '_labeled.csv')
            output_file = os.path.join(output_dir, output_filename)
            
            # Process file
            count = labeller.process_tokenized_file(input_file, output_file, category, label)
            total_articles += count
            
            # Read the created dataset for combining later
            df = pd.read_csv(output_file)
            all_datasets.append(df)
            
            # Move processed file to done folder
            done_file = os.path.join(done_dir, filename)
            shutil.move(input_file, done_file)
            print(f"Moved {filename} to tokenized/done/")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    # Create combined dataset
    if all_datasets:
        combined_df = pd.concat(all_datasets, ignore_index=True)
        combined_file = os.path.join(output_dir, "combined_labeled_dataset.csv")
        combined_df.to_csv(combined_file, index=False, encoding='utf-8')
        
        print(f"\nLabelling complete!")
        print(f"Total labeled articles: {total_articles}")
        print(f"Combined dataset: {combined_file}")
        print(f"Dataset shape: {combined_df.shape}")
        print(f"Label distribution:")
        print(combined_df['category'].value_counts())
        print(f"Files ready for model training!")
    else:
        print("No datasets were created successfully.")

if __name__ == "__main__":
    main()