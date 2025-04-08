import json
import random
import numpy as np
import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and preprocessing tools
model = load_model("chatbot_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    lbl_encoder = pickle.load(f)

# Load intents
with open("intents.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Chat function
def chat():
    print("🤖 Chatbot is ready! Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            print("👋 Goodbye!")
            break

        seq = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(seq, truncating='post', maxlen=20)

        predictions = model.predict(padded)[0]
        class_index = np.argmax(predictions)
        class_label = lbl_encoder.inverse_transform([class_index])[0]

        # Get a random response from matching intent
        for intent in data["intents"]:
            if intent["tag"] == class_label:
                print("Bot:", random.choice(intent["responses"]))
                break

if __name__ == "__main__":
    chat()
