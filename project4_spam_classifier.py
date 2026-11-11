from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Small example dataset (label, message)
data = [
    ("ham",  "Hey, are we still on for lunch tomorrow?"),
    ("ham",  "Can you send me the report by tonight?"),
    ("ham",  "Happy birthday! Hope you have a great day."),
    ("ham",  "Don't forget the meeting at 3pm."),
    ("spam", "WINNER!! You've won a free iPhone, click here now!"),
    ("spam", "Congratulations, you've been selected for a $1000 prize!"),
    ("spam", "URGENT: Your account will be suspended, verify now!"),
    ("spam", "Get cheap loans instantly, no credit check needed!"),
    ("ham",  "Mom called, she wants you to call her back."),
    ("spam", "Claim your free gift card by clicking this link!"),
    ("ham",  "The project deadline got moved to Friday."),
    ("spam", "You have won a lottery of $5000, reply to claim."),
]

labels = [item[0] for item in data]
messages = [item[1] for item in data]

# Step 1: Convert text into numeric vectors (word counts)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(messages)

# Step 2: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.3, random_state=42
)

# Step 3: Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 4: Test it
predictions = model.predict(X_test)
print("Predictions:", predictions)
print("Actual:     ", y_test)
print("Accuracy:", accuracy_score(y_test, predictions))

# Step 5: Try it on a brand new message
new_message = ["Click here to win a free vacation now!!!"]
new_vector = vectorizer.transform(new_message)
print("New message prediction:", model.predict(new_vector))
