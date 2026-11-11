from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Run this once. vader_lexicon is a dictionary NLTK downloads that
# maps thousands of words to sentiment scores (e.g., "great" = +3.1, "terrible" = -3.4).
# VADER's whole trick is looking up each word's score and combining them with some clever
# rules.
nltk.download('vader_lexicon')

# Create the analyzer object (loads the lexicon into memory)
sia = SentimentIntensityAnalyzer()

sentences = [
    "I love this product, it's amazing!",
    "This is the worst experience I've ever had.",
    "The package arrived on time.",
    "I don't think this is very good.",
    "NLP IS SO COOL!!!"
]

for sentence in sentences:
    scores = sia.polarity_scores(sentence)
    print(f"{sentence}")
    print(f"  -> {scores}")

    # Decide overall label using the compound score
    if scores['compound'] >= 0.05:
        label = "Positive"
    elif scores['compound'] <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"
    print(f"  -> Overall: {label}\n")
