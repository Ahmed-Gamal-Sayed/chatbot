import json
import random
import numpy as np
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
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

# Chatbot response function
def chatbot_response(user_input):
    seq = tokenizer.texts_to_sequences([user_input])
    padded = pad_sequences(seq, truncating='post', maxlen=20)

    predictions = model.predict(padded)[0]
    class_index = np.argmax(predictions)
    class_label = lbl_encoder.inverse_transform([class_index])[0]

    # Get a random response from matching intent
    for intent in data["intents"]:
        if intent["tag"] == class_label:
            return random.choice(intent["responses"])

# FastAPI App
app = FastAPI()

# Allow frontend access (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Schema
class ChatMessage(BaseModel):
    message: str

# API Route
@app.post("/chat")
async def chat_endpoint(msg: ChatMessage):
    response = chatbot_response(msg.message)
    return {"response": response}

@app.get('/')
async def root():
    return {'message': 'Startup server...'}
