import pandas as pd
from sklearn.feature_extraction.text import (  # type: ignore[reportMissingImports]
    CountVectorizer,
    TfidfVectorizer,
)

corpus = [
    "Machine learning is fascinating and deep.",
    "Deep learning is a subset of machine learning.",
    "Natural language processing uses machine learning."
]


print("=== 1. Bag of Words (Unigram) ===")
bow_vectorizer = CountVectorizer()
bow_matrix = bow_vectorizer.fit_transform(corpus)


df_bow = pd.DataFrame(
    bow_matrix.toarray(),
    columns=bow_vectorizer.get_feature_names_out(),
    index=["Doc 1", "Doc 2", "Doc 3"]
)
print(df_bow)


print("\n=== 2. N-grams (Bigrams: n=2) ===")
ngram_vectorizer = CountVectorizer(ngram_range=(2, 2))
ngram_matrix = ngram_vectorizer.fit_transform(corpus)

df_ngram = pd.DataFrame(
    ngram_matrix.toarray(),
    columns=ngram_vectorizer.get_feature_names_out(),
    index=["Doc 1", "Doc 2", "Doc 3"]
)
print(df_ngram)


print("\n=== 3. TF-IDF Representation ===")
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

df_tfidf = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=tfidf_vectorizer.get_feature_names_out(),
    index=["Doc 1", "Doc 2", "Doc 3"]
)
print(df_tfidf.round(3))
