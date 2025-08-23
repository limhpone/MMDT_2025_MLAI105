# Processes

## Scraping

- Three sources for three different categories (DVB, Myawady, Khitthit)

- Myanmar text detection used the Myanmar unicode block -> https://en.wikipedia.org/wiki/Myanmar_(Unicode_block)

- Removed html elements, hidden characters, replacing problematic characters

- Preserved Myanmar and ASCII characters

- multiple spaces -> single space, multiple newlines -> double newline

- After scraping produced .json file with contents as well as text files, however we only use .json file for training (the text files and other session related files were produced for debugging purposes)

- Always open with utf-8 encoding for Myanmar unicode texts

- The rest of the codes are for status printing as well as command line arguments so that we can specify how many articles to scrape.

## Data Preparation

- There are four processes
    1. Cleaning
    2. Preprocessing
    3. Tokenization
    4. Labelling 

### Cleaning

- normalize unicode "NFC(Normalization Form Canonical Composition)" -> visually similar mm texts can have different unicodes

- normalized white spaces, tabs, new lines to single space

- preserved these data in the text 
    - Myanmar script characters (U+1000–U+109F)

    - Basic ASCII printable characters (U+0020–U+007E)

    - Extended Latin-1 characters (U+00A0–U+00FF)

    - Myanmar Extended-A (U+AA60–U+AA7F)

    - Myanmar Extended-B (U+A9E0–U+A9FF)

    - General punctuation and currency symbols from Unicode blocks (U+2000–U+206F, U+2070–U+209F, U+20A0–U+20CF)

- sentence segmentation to prevent single long line output which is hard to manage while doing tokenization

### Preprocessing

- Prevent short and lengthy articles (currently 5 characters and 20,000 characters for each article)

- Remove duplicates (but temporarily disabled because it removed too much from Khitthit media articles: it used hashing of title and content for duplicate detection)

- Valide article structure (missing title or content)

### Tokenization

- This is the most important part and where a lot of errors occur.

- The restrictions on length of the input text and chuncking of it has to do here because I have encountered a lot of memory issue and slow tokenization speed during the tokenization processes.

- myWord from Sayar Ye Kyaw Thu is being used here. He used bigram and unigram dictionary to segment characters into the best tokenized Myanmar words segmentation. But it was too slow or my implementation was not efficient but I managed to improve the speed.

- Viterbi algorithms is used (but need to study further)

### Labelling

- For this we just put the categories for each tokenized files then create combined file for training

## Training

### Model's architecture

```python
model = Sequential([
    # Embedding layer
    Embedding(
        input_dim=self.vocab_size,
        output_dim=embedding_dim,
        input_length=self.max_length,
        mask_zero=True
    ),
    
    # Bidirectional LSTM layers
    Bidirectional(LSTM(lstm_units, return_sequences=True, dropout=dropout_rate)),
    Bidirectional(LSTM(lstm_units, dropout=dropout_rate)),
    
    # Dense layers
    Dense(64, activation='relu'),
    Dropout(dropout_rate),
    Dense(32, activation='relu'),
    Dropout(dropout_rate),
    
    # Output layer
    Dense(3, activation='softmax')  # 3 classes: red, neutral, green
])
```

- 80/20 train test split

```python
self.training_report['model_config'] = {
    'architecture': 'Bidirectional LSTM',
    'embedding_dim': embedding_dim,
    'lstm_units': lstm_units,
    'dropout_rate': dropout_rate,
    'vocab_size': self.vocab_size,
    'max_sequence_length': self.max_length,
    'num_classes': 3,
    'optimizer': 'adam',
    'loss_function': 'categorical_crossentropy'
}
```

```python
trainer.run_full_training(
    vocab_size=15000,      # Slightly smaller vocab for better generalization
    max_length=500,        # Reduced length for better performance  
    embedding_dim=256,     # Larger embeddings for better representation
    lstm_units=128,        # More units for better capacity
    dropout_rate=0.5,      # Higher dropout to prevent overfitting
    epochs=50,             # More epochs (early stopping will control)
    batch_size=64          # Larger batch size for stable gradients
)
```
```
token_count_category
0-500        11725
501-1000      2619
1001-1500      288
Name: count, dtype: int64
```

# Challenges in Project

## General Coding Challenges

- The whole vibe coding made me overcomplicate things and made me dependent on them most of the time.
- They produce reasonable working code however, fixing them is the most painful part of this whole process.
- Unfamiliarity with some of the python libaries as well as the based concept of this whole project also contribute to the time consuming part of the whole process.


## Web Scraping

- Data collection has been one of the easiest task as well as the hardest one so far.
- Scraping from Myawady cause me so much pain because the scraper initially caught some invisible unicode characters that are very hard to detect and clean them. This site also have very lengthy articles that make the scraper something crash. It also have English articles in between the Myanmar articles which made the scraper to break when I tried to scrape large set of articles.
