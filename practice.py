from nltk.sentiment import SentimentIntensityAnalyzer

import nltk
nltk.download('vader_lexicon')


sia = SentimentIntensityAnalyzer()


sentences = [
    "I absolutely love the new design of your website!",
    "The food was fresh, hot, and delicious.",
    "Customer service resolved my issue in minutes.",
    "The battery dies way too fast.",
    "This is the worst experience I have ever had.",
    "The package arrived completely broken and late.",
    "The package is out for delivery today.",
    "The meeting starts at two o'clock.",
    "The product is okay, but it could be better."
]

for sentence in sentences:
    scores = sia.polarity_scores(sentence)
    print(f"{sentence}")
    print(f"->{scores}")

    if scores['compound'] >= 0.05:
        label = "Positive"
    elif scores["compound"] <= 0.05:
        label = "Negative"
    else:
        label = "Neutral"

    print(f"  -> Overall: {label}\n")
