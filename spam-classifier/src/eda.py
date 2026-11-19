import re
from collections import Counter

import matplotlib.pyplot as plt

STOP_WORDS = {"i", "you", "to", "the", "a", "and", "in", "is", "it", "my",
              "me", "u", "of", "for", "your", "that", "have", "on", "are",
              "be", "at", "so", "but", "not", "can", "will", "do", "if",
              "we", "or", "with", "just", "get", "ur", "this", "when", "up",
              "what", "go", "all", "out", "no", "now", "was", "ok"}


def add_basic_features(df):
    df = df.copy()
    df["char_count"] = df["text"].str.len()
    df["word_count"] = df["text"].str.split().str.len()
    df["digit_count"] = df["text"].str.count(r"\d")
    return df


def plot_label_counts(df, save_path):
    counts = df["label"].value_counts()
    counts.plot(kind="bar", color=["blue", "orange"])
    plt.title("Ham vs Spam")
    plt.ylabel("Number of messages")
    plt.savefig(save_path)
    plt.close()


def plot_feature(df, column, save_path):
    plt.hist(df[df["label"] == "ham"][column], bins=30, alpha=0.6,
             label="ham", density=True, color="blue")
    plt.hist(df[df["label"] == "spam"][column], bins=30, alpha=0.6,
             label="spam", density=True, color="orange")
    plt.title(column + " by label")
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def most_common_words(df, label, n=10):
    words = []
    for text in df[df["label"] == label]["text"]:
        for word in re.findall(r"[a-z]+", text.lower()):
            if word not in STOP_WORDS and len(word) > 1:
                words.append(word)
    return Counter(words).most_common(n)
