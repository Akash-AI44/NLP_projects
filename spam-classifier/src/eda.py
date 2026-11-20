import re
import numpy as np
import pandas as pd
from collections import Counter

import matplotlib.pyplot as plt
from matplotlib.container import BarContainer

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


def plot_feature(df, column, save_path, title, bin_width=5):
    upper = df[column].quantile(0.99)
    bins = np.arange(0, upper + bin_width, bin_width)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True, sharey=True)
    for ax, label, color in zip(axes, ["ham", "spam"], ["tab:blue", "tab:orange"]):
        values = df[df["label"] == label][column]
        weights = np.ones(len(values)) / len(values) * 100
        ax.hist(values, bins=bins, weights=weights, color=color)

        median = values.median()
        ax.axvline(median, color="black", linestyle="--")
        ax.text(median, ax.get_ylim()[1] * 0.9, f"  median = {median:.0f}")

        ax.set_title(f"{label} messages")
        ax.set_xlabel(column.replace("_", " "))
        ax.set_xlim(0, upper)

    axes[0].set_ylabel("% of messages")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_buckets(df, column, edges, labels, save_path, title, xlabel):
    buckets = pd.cut(df[column], bins=edges,
                     labels=labels, include_lowest=True)
    table = pd.crosstab(buckets, df["label"], normalize="columns") * 100

    ax = table.plot(kind="bar", color=["tab:blue", "tab:orange"],
                    figsize=(8, 4), rot=0)
    for container in ax.containers:
        if isinstance(container, BarContainer):
            ax.bar_label(container, fmt="%.0f%%")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("% of messages in that class")
    ax.legend(title="")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def most_common_words(df, label, n=10):
    words = []
    for text in df[df["label"] == label]["text"]:
        for word in re.findall(r"[a-z]+", text.lower()):
            if word not in STOP_WORDS and len(word) > 1:
                words.append(word)
    return Counter(words).most_common(n)
