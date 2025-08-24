# 🤖 Multi-Model Training Framework

A comprehensive framework for training and comparing multiple neural network architectures for Myanmar text classification.

## 🏗️ Architecture Overview

### Mother Script: `multi_model_trainer.py`
- **Handles**: Data loading, preprocessing, coordination of all experiments
- **Features**: Interactive menu, progress tracking, result aggregation
- **Data Source**: `../data/15000_article/combined_labeled_dataset.csv` (14,989 samples)

### Individual Model Scripts (in `model_architectures/`)
- **Focus**: Pure architecture definition, hyperparameter tuning, training strategies
- **Data Input**: Pre-processed data from mother script
- **Output**: Training results and model artifacts

## 🎯 Available Model Architectures

### 1. Enhanced BiLSTM (`enhanced_bilstm.py`)
- **Features**: Attention mechanism, batch normalization, regularization
- **Hyperparameter Optimization**: Optuna-based tuning (30 trials)
- **Key Components**:
  - Bidirectional LSTM layers (2 layers)
  - Multi-head attention
  - Dense layers with dropout
  - Advanced callbacks (EarlyStopping, ReduceLROnPlateau)
- **Est. Training Time**: 30-45 minutes

### 2. BiLSTM + CRF (`bilstm_crf.py`)
- **Features**: Conditional Random Fields for sequence labeling
- **Use Case**: Better for sequence prediction tasks
- **Components**: BiLSTM → Dense → CRF layer
- **Est. Training Time**: 45-60 minutes

### 3. CNN-BiLSTM Hybrid (`cnn_bilstm_hybrid.py`)
- **Features**: CNN for local features + BiLSTM for sequences
- **Architecture**: Multiple CNN filters (2,3,4,5) + BiLSTM
- **Benefits**: Captures both local patterns and long-term dependencies
- **Est. Training Time**: 30-40 minutes

### 4. DistilBERT Classifier (`distilbert_classifier.py`)
- **Features**: Pre-trained multilingual DistilBERT
- **Benefits**: Transfer learning, multilingual support
- **Requirements**: `transformers` library
- **Est. Training Time**: 60-90 minutes

### 5. Lightweight Transformer (`transformer_encoder.py`)
- **Features**: Custom transformer implementation
- **Components**: Multi-head attention, position embeddings
- **Benefits**: Attention mechanisms, parallelizable
- **Est. Training Time**: 45-60 minutes

### 6. Ensemble Model (`ensemble_model.py`)
- **Features**: Combines predictions from multiple models
- **Methods**: Majority voting, weighted ensemble
- **Requirements**: Other models must be trained first
- **Est. Training Time**: 20-30 minutes

## 🚀 Usage

### Quick Start
```bash
cd 3_trainer/trainer
python multi_model_trainer.py
```

### Menu Options
- **A**: Train all models (4-6 hours total)
- **S**: Select specific models to train
- **C**: Create architecture scripts only
- **Q**: Quit

### Individual Model Training
```bash
cd 3_trainer/trainer/model_architectures
python enhanced_bilstm.py  # Train specific model
```

## 📊 Dataset Information

- **Source**: `data/15000_article/combined_labeled_dataset.csv`
- **Total Samples**: 14,989 articles
- **Classes**: 
  - Red (political): 4,998 samples
  - Neutral (general): 4,993 samples  
  - Green (military): 4,998 samples
- **Vocabulary Size**: 29,158 unique tokens
- **Max Sequence Length**: 6,841 tokens
- **Average Length**: 399.4 tokens

## 🔧 Installation

### Install Dependencies
```bash
pip install -r requirements_multi_model.txt
```

### Key Dependencies
- `tensorflow>=2.13.0` - Deep learning framework
- `optuna>=3.0.0` - Hyperparameter optimization
- `tensorflow-addons>=0.20.0` - CRF layers
- `transformers>=4.20.0` - BERT models
- `scikit-learn>=1.0.0` - ML utilities

## 📈 Model Performance Comparison

The framework automatically generates comprehensive reports comparing:

### Metrics Tracked
- **Test Accuracy**: Final model accuracy
- **Training Time**: Duration for each model
- **Hyperparameters**: Best parameters found
- **Classification Reports**: Per-class performance

### Output Files
- `enhanced_bilstm_results_TIMESTAMP.json`
- `bilstm_crf_results_TIMESTAMP.json`
- `cnn_bilstm_results_TIMESTAMP.json`
- `distilbert_results_TIMESTAMP.json`
- `transformer_results_TIMESTAMP.json`
- `ensemble_results_TIMESTAMP.json`

## 🎛️ Hyperparameter Tuning

### Enhanced BiLSTM Parameters
```python
{
    'embedding_dim': [64, 128, 256],
    'lstm1_units': [32, 64, 128],
    'lstm2_units': [16, 32, 64],
    'learning_rate': [1e-4, 1e-3, 1e-2],
    'dropout_rates': [0.1, 0.4],
    'attention_heads': [2, 4, 8],
    'use_second_lstm': [True, False],
    'use_attention': [True, False]
}
```

### Optimization Strategy
- **Method**: Optuna Tree-structured Parzen Estimator
- **Trials**: 30 per model
- **Objective**: Maximize validation accuracy
- **Early Stopping**: Prevents overfitting

## 🔍 Advanced Features

### Data Preprocessing
- **Automatic Tokenization**: Handles pre-tokenized Myanmar text
- **Vocabulary Building**: Creates word-to-index mappings
- **Sequence Padding**: Handles variable-length sequences
- **Class Balancing**: Computes class weights

### Training Enhancements
- **Early Stopping**: Prevents overfitting
- **Learning Rate Scheduling**: Adaptive learning rates
- **Model Checkpointing**: Saves best models
- **Progress Tracking**: Real-time training monitoring

### Result Analysis
- **Session-based Organization**: Timestamped result directories
- **Comprehensive Reports**: JSON and markdown formats
- **Model Comparison**: Side-by-side performance analysis
- **Error Analysis**: Detailed classification reports

## 🎯 Recommended Model Combinations

### For Best Accuracy
1. **Enhanced BiLSTM** (optimized hyperparameters)
2. **DistilBERT** (transfer learning)
3. **Ensemble** (combining top models)

### For Speed vs Accuracy
1. **CNN-BiLSTM Hybrid** (fast, good performance)
2. **Lightweight Transformer** (parallel processing)
3. **Enhanced BiLSTM** (highest accuracy)

### For Production Deployment
1. **Enhanced BiLSTM** (optimized, reliable)
2. **CNN-BiLSTM Hybrid** (good speed/accuracy balance)
3. **Ensemble** (best overall performance)

## 🚨 Troubleshooting

### Common Issues

**Memory Errors**:
- Reduce batch size in model scripts
- Cap max_length to 500 tokens
- Use gradient checkpointing

**CUDA Issues**:
- Ensure TensorFlow-GPU compatibility
- Check CUDA/cuDNN versions
- Use CPU fallback if needed

**Missing Dependencies**:
```bash
pip install tensorflow-addons transformers optuna
```

**Data Loading Issues**:
- Verify CSV file exists at correct path
- Check file permissions
- Ensure proper encoding (UTF-8)

## 📁 File Structure
```
3_trainer/trainer/
├── multi_model_trainer.py          # Mother script
├── requirements_multi_model.txt     # Dependencies
├── model_architectures/             # Individual models
│   ├── enhanced_bilstm.py
│   ├── bilstm_crf.py
│   ├── cnn_bilstm_hybrid.py
│   ├── distilbert_classifier.py
│   ├── transformer_encoder.py
│   └── ensemble_model.py
├── multi_model_reports/            # Generated results
└── data_cache.pkl                  # Cached preprocessed data
```

## 🎉 Next Steps

After running the framework:

1. **Analyze Results**: Compare model performances
2. **Select Best Model**: Choose highest accuracy model
3. **Deploy Model**: Use best model in production
4. **Ensemble Creation**: Combine multiple models
5. **Further Tuning**: Refine hyperparameters of best models

## 🤝 Contributing

To add new model architectures:

1. Create new script in `model_architectures/`
2. Follow the pattern of existing scripts
3. Add model info to `available_models` dict
4. Test with sample data

---

**Happy Training! 🚀**