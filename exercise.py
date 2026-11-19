
import nltk
import re

text = """
I love Machine Learning.
NLP is very interesting!
I am learning NLP with Python.
"""

# Step 1: Sentence Tokenization
sentences = nltk.sent_tokenize(text)

# Step 2: Word Tokenization
words = nltk.word_tokenize(text)

# Step 3: Lowercase
lowercase_words = [word.lower() for word in words]

# Step 4: Extract only words
clean_words = re.findall(r"\b\w+\b", text.lower())

# Step 5: Vocabulary
vocabulary = set(clean_words)

# Step 6: Statistics
print("Sentences:", sentences)
print("Word Tokens:", words)
print("Lowercase Tokens:", lowercase_words)
print("Clean Words:", clean_words)
print("Vocabulary:", vocabulary)
print("Total Tokens:", len(clean_words))
print("Vocabulary Size:", len(vocabulary))
