# Myanmar News Classification BiLSTM Pipeline

A comprehensive machine learning pipeline for classifying Myanmar news articles into sentiment categories (Red/Neutral/Green) using Bidirectional LSTM neural networks.

## Project Structure

```
redGreen/
├── 0_walkthrough/              # Jupyter notebooks for data exploration
├── 1_scrapers/                 # Web scraping tools
│   ├── cleaners/              # Unicode cleaning utilities
│   ├── advanced_scraper.py    # Main scraping orchestrator
│   ├── dvb_scraper.py         # DVB News scraper
│   ├── khitthit_scraper.py    # Khitthit News scraper
│   └── myawady_scraper.py     # Myawady News scraper
├── 2_processor/               # Data processing pipeline
│   ├── cleaner/               # Data cleaning utilities
│   ├── preprocessor/          # Text preprocessing
│   ├── tokenizer/             # Myanmar text tokenization
│   │   └── myWord/            # Myanmar word segmentation
│   └── labeller/              # Sentiment labeling
├── 3_trainer/                 # Model training components
│   ├── trainer/               # BiLSTM training scripts
│   └── output_model/          # Trained models and artifacts
├── 4_analyzer/                # Article analysis and testing
│   ├── article_analyzer.py   # Article classification
│   └── analysis_outputs/     # Analysis results
├── bilstm_pipeline.py         # Main pipeline orchestrator
├── run_streamlit.py           # Streamlit app launcher
├── streamlit_app.py           # Web interface
├── utils.py                   # Utility functions
└── requirements.txt           # Python dependencies
```

## Features

- **Multi-source Scraping**: Automated collection from DVB, Khitthit, and Myawady news sources
- **Myanmar Text Processing**: Specialized tokenization using MyWord segmentation
- **Sentiment Classification**: Three-class classification (Red/Neutral/Green)
- **BiLSTM Architecture**: Deep learning model optimized for Myanmar text
- **Web Interface**: Real-time classification via Streamlit
- **Complete Pipeline**: End-to-end automation from scraping to analysis

## Installation

1. Clone or copy the project files to your desired location
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### Main Pipeline

Run the main pipeline script:
```bash
python bilstm_pipeline.py
```

Available options:
- **1. Scraping** - Collect news articles from websites
- **2. Data Preparation** - Complete processing pipeline (clean → preprocess → tokenize → label)
- **3. Model Training** - Train BiLSTM classification model
- **4. Analysis** - Analyze articles with trained model
- **9. Cleanup** - Clean up processed data folders
- **0. Exit**

### Web Interface

Launch the Streamlit web application:
```bash
python run_streamlit.py
```
or
```bash
streamlit run streamlit_app.py
```

## Usage Workflow

1. **Scraping**: Collect articles from Myanmar news sources
2. **Data Preparation**: Process raw articles through cleaning, preprocessing, tokenization, and labeling
3. **Model Training**: Train the BiLSTM model on processed data
4. **Analysis**: Test the model on new articles

## Data Flow

```
Raw Articles → Cleaned → Preprocessed → Tokenized → Labeled → Training → Model → Analysis
```

## Model Details

- **Architecture**: Bidirectional LSTM with embedding layer
- **Input**: Myanmar text tokenized using MyWord segmentation
- **Output**: Three classes (Red/Neutral/Green sentiment)
- **Training**: Uses labeled dataset with class balancing

## Requirements

- Python 3.7+
- TensorFlow 2.x
- Streamlit
- Myanmar text processing libraries
- Web scraping dependencies

## File Organization

The pipeline automatically organizes data into structured folders:
- `data/raw/` - Original scraped articles
- `data/cleaned/` - Cleaned text data
- `data/preprocessed/` - Preprocessed articles
- `data/tokenized/` - Tokenized text
- `data/labelled/` - Sentiment-labeled dataset

## Troubleshooting

### Common Issues

1. **MyWord Tokenizer Issues**: Ensure all dictionary files are in `2_processor/tokenizer/myWord/resources/`
2. **Import Errors**: Check that all paths are correctly set in the pipeline
3. **Memory Issues**: Reduce batch size or article count for training
4. **Scraping Failures**: Check internet connection and website availability

### Debug Mode

Enable verbose output by checking console logs during pipeline execution.

## License

This project is for educational purposes. Please respect the terms of service of news websites when scraping.

## Contributing

1. Follow the existing code structure
2. Test all pipeline components
3. Update documentation for new features
4. Ensure MyWord tokenizer compatibility

## Contact

For issues or questions, please refer to the project documentation or contact the development team.