
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize


from collections import Counter
import string


nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

text = """
Natural language processing is a fascinating field. NLP allows computers to understand, interpret, and generate human language. Many companies use
NLP for chatbots, translation, and sentiment analysis.
"""

sentences = sent_tokenize(text)
print("Sentences:", sentences)
# Step 1: Lowercase everything
text = text.lower()

# Step 2: Tokenize — split text into individual words
tokens = word_tokenize(text)
print("Tokens:", tokens[:10])


# Step 3: Remove punctuation
tokens = [word for word in tokens if word not in string.punctuation]


# Step 4: Remove stopwords (common words like "is", "the", "a")
# returns a list of ~179 common English words.
stop_words = set(stopwords.words('english'))
tokens = [word for word in tokens if word not in stop_words]
print("Tokens:", tokens[:10])

# Step 5: Count word frequency
word_freq = Counter(tokens)
print("Most common words:", word_freq.most_common(5))


stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

print("\n--- Stemming vs Lemmatization ---")

for word in tokens:
    stemmed = stemmer.stem(word)
    lemmatized = lemmatizer.lemmatize(word, pos='v')
    print(
        f"Original: {word:<10} | Stemmed: {stemmed:<10} | Lemmatized: {lemmatized}")
