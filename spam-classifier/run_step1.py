from src.config import PROCESSED_DATA_PATH
from src.data_loader import load_raw_data, clean_data


raw_df = load_raw_data()
print("RAW shape:", raw_df.shape)
print("RAW columns:", raw_df.columns.tolist())

df = clean_data(raw_df)
print("\nCLEAN shape:", df.shape)
print(df.head())

print("\nClass counts:")
print(df["label"].value_counts())
print("\nClass percentages:")
print((df["label"].value_counts(normalize=True) * 100).round(2))

df.to_csv(PROCESSED_DATA_PATH, index=False)
print("\nSaved to:", PROCESSED_DATA_PATH)
