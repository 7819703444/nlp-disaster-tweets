#!/usr/bin/env python
# coding: utf-8

# # Week 4: NLP Disaster Tweets Kaggle Mini-Project
# ## Fateme Hoshyar Zare
# 
# # 1. Brief Description of the Problem and Data
# ## 1-1. Introduction
# Social media platforms like Twitter have become crucial channels for real-time communication during emergencies.
# This project aims to build a machine learning model that can predict whether a tweet is related to a disaster or not.
# 
# # 1-2. Import Libraries
# We'll start by importing the necessary Python libraries.
# 

# In[5]:


# Basic libraries
import os
import numpy as np
import pandas as pd

# Libraries for visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Libraries for text processing
import string
import re

# Library for natural language processing (NLP)
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Library for splitting dataset
from sklearn.model_selection import train_test_split

# Libraries for neural networks
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout

# Library for evaluation
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, precision_score

import warnings
warnings.filterwarnings("ignore")


# # 1-3. Import Data
# First, we will check the data files and import them. Next, we will take a look at the data.

# In[6]:


# Provide the local paths to the datasets
train_data_path = r"C:\Users\farza\Downloads\Fateme\nlp-getting-started\train.csv"
test_data_path = r"C:\Users\farza\Downloads\Fateme\nlp-getting-started\test.csv"

# Load datasets
train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

# Display samples from the training data
print(train_data.sample(5))

# The training data includes 'id,' 'keyword,' 'location,' 'text,' and 'target.'
# The target column indicates whether a tweet is related to a disaster (1) or not (0).

# Display samples from the test data
print(test_data.sample(5))

# Print the shape of the datasets
print("Train data: Number of rows =", train_data.shape[0], ", Number of columns =", train_data.shape[1])
print("Test data: Number of rows =", test_data.shape[0], ", Number of columns =", test_data.shape[1])


# In[7]:


# 2. Exploratory Data Analysis (EDA)
# 2-1. Target Distribution
# We will examine the distribution of the 'target' in the training data.

plt.figure(figsize=(5,3))
colors = ["blue", "red"]

sns.countplot(x='target', data=train_data, palette=colors)
plt.title('Target Distribution \\n (0: Non-Disaster || 1: Disaster)', fontsize=14)
plt.show()

print(train_data['target'].value_counts())


# In[8]:


# 2-2. Top 20 Keywords
# We will identify the top 20 keywords for both 'Non-disaster' and 'Disaster' tweets.

nondisaster_keywords = train_data.loc[train_data["target"] == 0]["keyword"].value_counts()
disaster_keywords = train_data.loc[train_data["target"] == 1]["keyword"].value_counts()

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
sns.barplot(y=nondisaster_keywords[0:20].index, x=nondisaster_keywords[0:20], orient='h', ax=ax[0], palette="Blues_d")
ax[0].set_title("Top 20 Keywords - Non-Disaster Tweets")
ax[0].set_xlabel("Keyword Frequency")

sns.barplot(y=disaster_keywords[0:20].index, x=disaster_keywords[0:20], orient='h', ax=ax[1], palette="Reds_d")
ax[1].set_title("Top 20 Keywords - Disaster Tweets")
ax[1].set_xlabel("Keyword Frequency")

plt.tight_layout()
plt.show()


# In[9]:


# 2-3. Text Length Distribution
# We will analyze the distribution of text lengths for both 'Non-disaster' and 'Disaster' tweets.

train_data["length"] = train_data["text"].apply(len)

# Filter the data for target = 0 and target = 1
target_0_data = train_data[train_data["target"] == 0]
target_1_data = train_data[train_data["target"] == 1]

# Create subplots for side-by-side histograms
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Plot the histogram for target = 0
sns.histplot(target_0_data["length"], color='blue', bins=30, ax=ax1)
ax1.set_title("Length of Non-disaster Tweets")
ax1.set_xlabel("Number of Characters")
ax1.set_ylabel("Frequency")
ax1.set_ylim(0, 700)

# Plot the histogram for target = 1
sns.histplot(target_1_data["length"], color='red', bins=30, ax=ax2)
ax2.set_title("Length of Disaster Tweets")
ax2.set_xlabel("Number of Characters")
ax2.set_ylabel("Frequency")
ax2.set_ylim(0, 700)

plt.show()


# In[10]:


# 2-4. Text Preprocessing
# We will clean the text data by removing punctuation, URLs, HTML tags, non-ASCII characters, and stopwords.
# We will also replace abbreviations with their full forms.

# Function to clean text
def clean_text(text):
    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    
    # Remove URLs
    text = re.sub(r'https?://\\S+|www\\.\\S+', 'URL', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove non-ASCII characters
    text = ''.join([char for char in text if char in string.printable])
    
    return text

# Apply text cleaning
train_data['clean_text'] = train_data['text'].apply(clean_text)
print(train_data.head())

# We will now remove stopwords using the NLTK library.

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word.lower() not in stop_words])

train_data['clean_text'] = train_data['clean_text'].apply(remove_stopwords)
print(train_data.head())


# In[11]:


# 3. Model Architecture
# 3-1. Splitting the Train and Validation Data
# We will split the training data into 80% training and 20% validation sets.

y = train_data['target']
X_train, X_valid, y_train, y_valid = train_test_split(train_data['clean_text'], y, test_size=0.2, random_state=42)

# 3-2. Tokenization and Padding
# We will tokenize the text data and pad the sequences to ensure uniform input length.

max_features = 3000
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_valid = tokenizer.texts_to_sequences(X_valid)

X_train = pad_sequences(X_train)
X_valid = pad_sequences(X_valid)

# 3-3. Building and Training the LSTM Model
# We will build an LSTM model and train it using the training data.

embed_dim = 32
lstm_out = 32

def build_model(embed_dim, lstm_out):
    model = Sequential()
    model.add(Embedding(max_features, embed_dim, input_length=X_train.shape[1]))
    model.add(Dropout(0.2))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.4))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = build_model(embed_dim, lstm_out)
model.summary()

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_valid, y_valid))


# In[12]:


# 4. Results and Analysis
# 4-1. Evaluation
# We will evaluate the model using accuracy, recall, and precision.

y_pred = model.predict(X_valid).round()

print('Accuracy:', accuracy_score(y_valid, y_pred))
print('Recall:', recall_score(y_valid, y_pred))
print('Precision:', precision_score(y_valid, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_valid, y_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='cool')
plt.show()

# Classification Report
print(classification_report(y_valid, y_pred))


# # 5. Conclusion
# In this project, we built an LSTM model to predict whether a tweet is related to a disaster or not.
# #The model achieved reasonable accuracy, recall, and precision, indicating its effectiveness in this task.
# 
# Future improvements can be made by experimenting with different model architectures, hyperparameters, and more advanced text preprocessing techniques.
# 
# # References
# Addison Howard, devrishi, Phil Culliton, Yufeng Guo. (2019). Natural Language Processing with Disaster Tweets.
# https://www.kaggle.com/c/nlp-getting-started
# 

# In[ ]:




