import pandas as pd

from src.config import PROCESSED_DATA_PATH, PROJECT_ROOT
from src.eda import (add_basic_features, plot_label_counts, plot_feature,
                     plot_buckets, most_common_words)

reports = PROJECT_ROOT / "reports"

df = pd.read_csv(PROCESSED_DATA_PATH)
df = add_basic_features(df)

plot_label_counts(df, reports / "01_label_counts.png")
plot_feature(df, "char_count", reports / "02_char_count.png",
             "Spam messages are longer: about 150 characters vs about 50",
             bin_width=10)

plot_buckets(df, "digit_count", [-1, 0, 5, 10, 20, 1000],
             ["0", "1-5", "6-10", "11-20", "21+"],
             reports / "03_digit_count.png",
             "Most ham has no digits; most spam has 10 or more",
             "Number of digits in the message")

print(df.groupby("label")[["char_count", "word_count", "digit_count"]].mean())
print("Top spam words:", most_common_words(df, "spam"))
print("Top ham words:", most_common_words(df, "ham"))
