# Natural Language Processing Projects

This folder contains a set of beginner-friendly Python projects focused on core NLP concepts and text-processing techniques. Each script demonstrates a different idea from the natural language processing pipeline, from cleaning raw text to building simple language-aware applications.

## Project Overview

### 1. Text Preprocessing
File: `Text_preprocessing.py`

This project demonstrates the basic text-cleaning workflow used in NLP:
- Lowercasing
- Tokenization
- Removing punctuation
- Removing stopwords
- Frequency counting
- Stemming and lemmatization

It uses NLTK to process sample text and shows how raw language data is transformed into more useful features for downstream analysis.

### 2. Bag of Words, N-Grams, and TF-IDF
File: `BoWNgramsTFIDF.py`

This script explores common text representation methods:
- Bag of Words (unigrams)
- N-grams (bigrams)
- TF-IDF weighting

It creates a small example corpus and converts text into numerical vectors that machine learning models can understand.

### 3. Sentiment Analysis
Files:
- `practice.py`
- `project2_sentiment.py`

These scripts use VADER sentiment analysis from NLTK to classify sentences as positive, negative, or neutral based on their emotional tone.

They help demonstrate how sentiment scores can be derived from words such as "love", "worst", or "amazing" and then summarized into an overall sentiment label.

### 4. Named Entity Recognition (NER)
File: `project3_ner.py`

This project uses spaCy's pre-trained English model to identify named entities in text, such as:
- People
- Organizations
- Locations
- Dates

It prints the extracted entities and visualizes them with spaCy's displaCy output.

### 5. Spam Classifier
File: `project4_spam_classifier.py`

This project builds a simple email spam detector using:
- CountVectorizer for text-to-feature conversion
- Train/test splitting
- Multinomial Naive Bayes classifier
- Accuracy evaluation

It shows a basic supervised learning workflow for text classification.

### 6. Simple Chatbot
File: `project5_chatbot.py`

This script creates a small rule-based chatbot that responds to common intents such as:
- Greeting
- Goodbye
- Thanks
- Name inquiry

The chatbot uses keyword matching and predefined responses rather than full machine learning or deep learning.

## Technologies Used

- Python
- NLTK
- scikit-learn
- spaCy
- pandas
- Regular expressions and basic NLP preprocessing

## Setup

Make sure you have Python installed, then install the required packages:

```bash
pip install pandas scikit-learn nltk spacy
```

For the spaCy NER example, download the English model:

```bash
python -m spacy download en_core_web_sm
```

You may also need to download NLTK datasets when running the preprocessing and sentiment scripts:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
```

## How to Run

Run each project individually with Python, for example:

```bash
python Text_preprocessing.py
python BoWNgramsTFIDF.py
python project2_sentiment.py
python project3_ner.py
python project4_spam_classifier.py
python project5_chatbot.py
```

## Learning Goals

These projects are designed to help learners understand:
- How text is processed before modeling
- How words are converted into numeric representations
- How sentiment and entity detection work
- How basic text classifiers are trained
- How simple conversational bots are built

## Summary

This collection provides a practical introduction to natural language processing using Python. It covers both core NLP fundamentals and beginner-level machine learning applications for language data.
