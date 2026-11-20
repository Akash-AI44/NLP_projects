import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

STOP_WORDS = set(stopwords.words("english"))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    text = text.lower()
    text = re.sub(r"&\w+;", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " urltoken ", text)
    text = re.sub(r"[£$€]", " currtoken ", text)
    text = re.sub(r"\d+", " numtoken ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords(tokens):
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]


def stem_tokens(tokens):
    return [stemmer.stem(t) for t in tokens]


def lemmatize_tokens(tokens):
    return [lemmatizer.lemmatize(t) for t in tokens]


def preprocess(text, method="lemma"):
    tokens = clean_text(text).split()
    tokens = remove_stopwords(tokens)
    if method == "stem":
        tokens = stem_tokens(tokens)
    elif method == "lemma":
        tokens = lemmatize_tokens(tokens)
    return " ".join(tokens)
