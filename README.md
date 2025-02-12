# Sentiment Analysis of Rotten Tomatoes Movie Reviews

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Methodology](#methodology)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Model Saving & Loading](#model-saving--loading)
- [Contributors](#contributors)
- [License](#license)

## Project Overview
This project compares traditional machine learning and deep learning approaches for sentiment analysis using movie reviews from Rotten Tomatoes. We implement:
- Logistic Regression with TF-IDF features
- Bidirectional LSTM with GloVe embeddings

Key features:
- Comprehensive EDA with text visualization
- Advanced text preprocessing pipeline
- Hyperparameter tuning and class balancing
- Model performance comparison using multiple metrics

## Dataset Description
**Source**: Rotten Tomatoes movie reviews  
**Size**: 480,000 reviews (balanced 240k fresh/rotten)  
**Columns**:
- `Review`: Text of critic reviews
- `Freshness`: Sentiment label (fresh=1/rotten=0)

[Dataset Link](https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset)

## Methodology

### Preprocessing Pipeline
1. Text cleaning (lowercasing, punctuation removal)
2. Stopword removal (preserving negations)
3. TF-IDF vectorization (for Logistic Regression)
4. Sequence padding (for LSTM)

### Model Architectures
| Model                | Key Features                                  |
|----------------------|----------------------------------------------|
| Logistic Regression  | TF-IDF features, L2 regularization           |
| Bidirectional LSTM   | GloVe embeddings, Spatial dropout, Class weights |

Models are saved here:

```
Models/
├── logistic_regression_model.pkl
└── lstm_model/
    ├── sentiment_lstm_model.keras
    ├── tokenizer.pkl
    └── model_params.json
```
### Load the models
```python

    from tensorflow.keras.models import load_model
    import pickle

    # Load Logistic Regression
    with open('Models/logistic_regression_model.pkl', 'rb') as f:
        lr_model = pickle.load(f)

    # Load LSTM components
    lstm_model = load_model('Models/lstm_model/sentiment_lstm_model.keras')
    with open('Models/lstm_model/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    # Example prediction
    sample_review = "This film combines brilliant acting with a weak script"
    print(predict_sentiment(sample_review))  # Output: Rotten

```

## Installation

### Requirements
```bash
Python 3.8+
pip install -r requirements.txt