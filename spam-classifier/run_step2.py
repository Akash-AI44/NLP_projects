import pandas as pd

from src.config import PROCESSED_DATA_PATH, PROJECT_ROOT
from src.eda import (add_basic_features, plot_label_counts,
                     plot_feature, most_common_words)

reports = PROJECT_ROOT / "reports"

df = pd.read_csv(PROCESSED_DATA_PATH)
df = add_basic_features(df)

plot_label_counts(df, reports / "01_label_counts.png")
plot_feature(df, "char_count", reports / "02_char_count.png")
plot_feature(df, "digit_count", reports / "03_digit_count.png")

print(df.groupby("label")[["char_count", "word_count", "digit_count"]].mean())
print("Top spam words:", most_common_words(df, "spam"))
print("Top ham words:", most_common_words(df, "ham"))
