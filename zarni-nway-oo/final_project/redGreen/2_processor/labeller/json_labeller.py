import json
import os
import shutil
import pandas as pd
import uuid

class MyanmarJSONLabeller:
    def __init__(self):
        """Initialize Myanmar JSON labeller"""
        self.label_mapping = {
            'red': 0,
            'neutral': 1, 
            'green': 2
        }
    
    def determine_label_from_filename(self, filename):
        """Determine label based on filename patterns"""
        filename_lower = filename.lower()
        
        if 'khitthit' in filename_lower or 'red' in filename_lower:
            return 'red', 0
        elif 'myawady' in filename_lower or 'green' in filename_lower:
            return 'green', 2
        elif 'dvb' in filename_lower or 'neutral' in filename_lower:
            return 'neutral', 1
        else:
            raise ValueError(f"Cannot determine label from filename: {filename}")
    
    def count_tokens(self, text):
        """Count tokens in tokenized text"""
        if not text:
            return 0
        
        # Count words by splitting on spaces
        tokens = text.split()
        return len(tokens)
    
    def process_tokenized_json(self, input_file, category, label):
        """Process a tokenized JSON file and create labeled dataset"""
        print(f"Labelling {category} articles from: {os.path.basename(input_file)}")
        
        # Load tokenized JSON
        with open(input_file, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        
        if not isinstance(articles, list):
            raise ValueError("JSON file should contain a list of articles")
        
        labeled_data = []
        total_articles = len(articles)
        
        print(f"  Processing {total_articles} articles...")
        
        for i, article in enumerate(articles):
            if not isinstance(article, dict):
                continue
                
            # Check required fields
            if 'title' not in article or 'content' not in article:
                continue
            
            title = article['title']
            content = article['content']
            
            if not title or not content:
                continue
            
            # Create record ID based on category and index
            record_id = f"{category[0]}_{i+1}"
            
            # Combine title and content as full article text
            # This maintains article-level consistency as you requested
            combined_tokens = f"{title} {content}"
            token_count = self.count_tokens(combined_tokens)
            
            # Create labeled record (only keep essential fields for training)
            labeled_record = {
                'id': record_id,
                'category': category,
                'label': label,
                'tokens': combined_tokens,
                'token_count': token_count,
                'title': title,  # Keep separate for reference
                'content': content  # Keep separate for reference
            }
            
            labeled_data.append(labeled_record)
            
            # Progress tracking
            if (i + 1) % 50 == 0 or (i + 1) == total_articles:
                progress = (i + 1) / total_articles * 100
                print(f"    Progress: {i + 1}/{total_articles} ({progress:.1f}%)")
        
        print(f"  Created {len(labeled_data)} labeled records")
        return labeled_data
    
    def save_individual_csv(self, labeled_data, output_file, category):
        """Save individual category CSV file"""
        if not labeled_data:
            print(f"No data to save for {category}")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(labeled_data)
        
        # Reorder columns for better readability
        column_order = ['id', 'category', 'label', 'tokens', 'token_count', 
                       'title', 'content']
        df = df[column_order]
        
        # Save to CSV
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"  Saved {len(df)} records to: {output_file}")
    
    def create_combined_dataset(self, all_labeled_data, output_file):
        """Create combined dataset from all categories"""
        if not all_labeled_data:
            print("No labeled data to combine")
            return
        
        # Flatten all labeled data
        combined_data = []
        for category_data in all_labeled_data:
            combined_data.extend(category_data)
        
        # Convert to DataFrame
        df = pd.DataFrame(combined_data)
        
        # Reorder columns for training (simplified format)
        training_columns = ['id', 'category', 'label', 'tokens', 'token_count']
        training_df = df[training_columns]
        
        # Save combined training dataset
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        training_df.to_csv(output_file, index=False, encoding='utf-8')
        
        # Print statistics
        print(f"\nCombined Dataset Statistics:")
        print(f"  Total articles: {len(training_df)}")
        print(f"  Label distribution:")
        for category in training_df['category'].unique():
            count = len(training_df[training_df['category'] == category])
            percentage = (count / len(training_df)) * 100
            print(f"    {category}: {count} ({percentage:.1f}%)")
        
        print(f"  Average tokens per article: {training_df['token_count'].mean():.1f}")
        print(f"  Token count range: {training_df['token_count'].min()}-{training_df['token_count'].max()}")
        print(f"  Combined dataset saved to: {output_file}")

def main():
    """Process all tokenized JSON files and create labeled datasets"""
    labeller = MyanmarJSONLabeller()
    
    # Use relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, '..', '..')
    input_dir = os.path.join(project_root, "data", "tokenized", "to_process")
    output_dir = os.path.join(project_root, "data", "labelled", "to_process")
    done_dir = os.path.join(project_root, "data", "tokenized", "done")
    
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(done_dir, exist_ok=True)
    
    # Process all JSON files in the to_process directory
    if not os.path.exists(input_dir):
        print(f"Input directory does not exist: {input_dir}")
        return
    
    files_to_process = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    
    if not files_to_process:
        print(f"No .json files found in: {input_dir}")
        return
    
    print(f"Found {len(files_to_process)} JSON files to label: {files_to_process}")
    
    all_labeled_data = []
    
    for filename in files_to_process:
        input_file = os.path.join(input_dir, filename)
        
        try:
            # Determine category and label from filename
            category, label = labeller.determine_label_from_filename(filename)
            
            # Process the tokenized JSON file
            labeled_data = labeller.process_tokenized_json(input_file, category, label)
            
            if labeled_data:
                # Save individual category CSV
                base_name = os.path.splitext(filename)[0]
                individual_csv = os.path.join(output_dir, f"{category}_{base_name}_labeled.csv")
                labeller.save_individual_csv(labeled_data, individual_csv, category)
                
                # Add to combined dataset
                all_labeled_data.append(labeled_data)
            
            # Move processed file to done
            done_file = os.path.join(done_dir, filename)
            shutil.move(input_file, done_file)
            print(f"Moved {filename} to tokenized/done/")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    # Create combined training dataset
    if all_labeled_data:
        combined_output = os.path.join(output_dir, "combined_labeled_dataset.csv")
        labeller.create_combined_dataset(all_labeled_data, combined_output)
    
    print(f"\nLabelling complete!")
    print(f"Individual and combined datasets ready in: {output_dir}")

if __name__ == "__main__":
    main()