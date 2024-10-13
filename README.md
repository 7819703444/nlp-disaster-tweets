

# NLP Disaster Tweets Kaggle Mini-Project

This project classifies tweets as either disaster-related or not using natural language processing (NLP) techniques. It utilizes the dataset provided by Kaggle's "Natural Language Processing with Disaster Tweets" competition.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Preprocessing](#preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)
- [Future Work](#future-work)
- [References](#references)

## Project Overview
The goal of this project is to build an LSTM-based machine learning model to classify tweets as related to real disasters or not. 

### Objective
To predict whether a tweet is about a real disaster (`1`) or not (`0`).

## Dataset
The dataset contains labeled tweets that are classified as either related to a disaster or not. It can be downloaded from the [Kaggle NLP Disaster Tweets competition page](https://www.kaggle.com/c/nlp-getting-started/data).

- **train.csv**: The dataset used for training the model, containing 7613 tweets.
- **test.csv**: The dataset for testing the model, containing 3263 tweets.

The target labels:
- `1` represents disaster-related tweets.
- `0` represents non-disaster tweets.

## Model Architecture
The model uses a sequential architecture with the following layers:
1. **Embedding Layer**: For word embeddings.
2. **LSTM Layer**: For capturing long-term dependencies in text.
3. **Dropout Layer**: To prevent overfitting.
4. **Dense Layer**: For final classification.

### Model Summary
```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding (Embedding)        (None, 30, 32)            96000     
lstm (LSTM)                  (None, 32)                8320      
dropout (Dropout)            (None, 32)                0         
dense (Dense)                (None, 1)                 33        
=================================================================
Total params: 104,353
Trainable params: 104,353
Non-trainable params: 0
_________________________________________________________________
```

## Preprocessing
1. **Text Cleaning**: Removing punctuation, URLs, non-ASCII characters, stopwords, and abbreviations.
2. **Tokenization**: Converting text into sequences using `Tokenizer`.
3. **Padding**: Padding sequences to ensure uniform input length.
4. **Lemmatization**: Converting words to their base forms.

## Exploratory Data Analysis
### Target Distribution
- The dataset contains more non-disaster-related tweets than disaster-related ones.
- Target distribution:
  - `0` (non-disaster): 4342 tweets.
  - `1` (disaster): 3271 tweets.

### Top 20 Keywords
Analyzed keywords that appeared in disaster and non-disaster tweets.

### Text Length Distribution
Examined the number of characters in both disaster and non-disaster tweets, visualizing the results using histograms.

## Training and Evaluation
The model was trained using an LSTM architecture. The training process was performed for **10 epochs**, with an 80-20 split between training and validation data.

### Training Hyperparameters:
- **Embedding Dimension**: 32
- **LSTM Output Units**: 32
- **Batch Size**: 32
- **Loss Function**: Binary Crossentropy
- **Optimizer**: Adam

## Results

The LSTM model achieved the following results on the validation set:
- **Accuracy**: 75.18%
- **Precision**: 71.61%
- **Recall**: 69.18%

### Confusion Matrix
```
[[699 175]
 [201 448]]
```

### Classification Report
```
              precision    recall  f1-score   support

           0       0.78      0.80      0.79       874
           1       0.72      0.69      0.70       649

    accuracy                           0.75      1523
   macro avg       0.75      0.74      0.75      1523
weighted avg       0.75      0.75      0.75      1523
```

## Future Work
- **Model Tuning**: Further experiment with hyperparameter tuning to improve the model.
- **Advanced Architectures**: Try advanced NLP models like BERT or transformer-based models.
- **Feature Engineering**: Explore more sophisticated text features or try word embeddings like GloVe or FastText.

## References
1. [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started)
2. Addison Howard, devrishi, Phil Culliton, Yufeng Guo. (2019). Kaggle NLP Disaster Tweets.
