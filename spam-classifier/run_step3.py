import pandas as pd

from src.config import PROCESSED_DATA_PATH, PROJECT_ROOT
from src.preprocessing import (clean_text, remove_stopwords, stem_tokens,
                               lemmatize_tokens, preprocess)

df = pd.read_csv(PROCESSED_DATA_PATH)

# 1. Follow ONE message through every stage
sample = df.loc[df["label"] == "spam", "text"].iloc[1]
print("ORIGINAL :", sample)
cleaned = clean_text(sample)
print("CLEANED  :", cleaned)
tokens = cleaned.split()
print("TOKENS   :", tokens)
no_stop = remove_stopwords(tokens)
print("NO STOP  :", no_stop)
print("STEMMED  :", stem_tokens(no_stop))
print("LEMMATIZED:", lemmatize_tokens(no_stop))

# 2. Preprocess every message in two ways
df["text_stem"] = df["text"].apply(lambda t: preprocess(t, method="stem"))
df["text_lemma"] = df["text"].apply(lambda t: preprocess(t, method="lemma"))

# 3. Compare vocabulary sizes


def vocab_size(series):
    return len(set(" ".join(series).split()))


print("\nUnique words, raw text   :", vocab_size(df["text"].str.lower()))
print("Unique words, cleaned    :", vocab_size(df["text"].apply(clean_text)))
print("Unique words, stemmed    :", vocab_size(df["text_stem"]))
print("Unique words, lemmatized :", vocab_size(df["text_lemma"]))

# 4. Messages that became empty after cleaning
print("\nEmpty after preprocessing:", (df["text_lemma"] == "").sum())

# 5. Save
out_path = PROJECT_ROOT / "processed" / "spam_preprocessed.csv"
df.to_csv(out_path, index=False)
print("Saved to:", out_path)
