import pandas as pd

from src.config import RAW_DATA_PATH


def load_raw_data(path=RAW_DATA_PATH):
    df = pd.read_csv(path, encoding="latin-1")
    return df


def clean_data(df):
    df = df.iloc[:, :2].copy()
    df.columns = ["label", "text"]

    df = df.dropna()
    df = df.drop_duplicates(subset="text")

    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

    df = df.reset_index(drop=True)
    return df
