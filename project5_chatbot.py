import random

# Dictionary mapping intents to possible user phrases and bot responses
responses = {
    "greeting": {
        "patterns": ["hi", "hello", "hey", "good morning"],
        "replies": ["Hello!", "Hi there!", "Hey, how can I help?"]
    },
    "goodbye": {
        "patterns": ["bye", "goodbye", "see you", "exit", "quit"],
        "replies": ["Goodbye!", "See you later!", "Take care!"]
    },
    "thanks": {
        "patterns": ["thanks", "thank you", "appreciate it"],
        "replies": ["You're welcome!", "No problem!", "Anytime!"]
    },
    "name": {
        "patterns": ["your name", "who are you"],
        "replies": ["I'm a simple chatbot built while learning NLP!"]
    }
}


def get_response(user_input):
    user_input = user_input.lower()

    for intent, data in responses.items():
        for pattern in data["patterns"]:
            if pattern in user_input:
                return random.choice(data["replies"])

    return "Sorry, I didn't understand that. Can you rephrase?"


# Main chat loop
print("Chatbot: Hi! Type 'exit' to end the chat.")
while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Chatbot:", get_response(user_input))
        break

    reply = get_response(user_input)
    print("Chatbot:", reply)
