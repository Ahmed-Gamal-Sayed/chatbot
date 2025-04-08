import json
import numpy as np
import random
import pickle

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load intents JSON
with open("intents.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Prepare data
training_sentences = []
training_labels = []
labels = []
responses = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        training_sentences.append(pattern)
        training_labels.append(intent["tag"])
    responses.append(intent["responses"])
    if intent["tag"] not in labels:
        labels.append(intent["tag"])

# Encode labels
lbl_encoder = LabelEncoder()
lbl_encoder.fit(training_labels)
training_labels_encoded = lbl_encoder.transform(training_labels)

# Tokenize words
vocab_size = 1000
embedding_dim = 16
max_len = 20
oov_token = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(training_sentences)
sequences = tokenizer.texts_to_sequences(training_sentences)
padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)

# Build model
model = Sequential()
model.add(Dense(128, input_shape=(max_len,), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(labels), activation='softmax'))

# Compile model
model.compile(loss='sparse_categorical_crossentropy', 
              optimizer=Adam(learning_rate=0.01), 
              metrics=['accuracy'])

# Train model
model.fit(padded_sequences, np.array(training_labels_encoded), epochs=200)

# Save model and tools
model.save("chatbot_model.h5")
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(lbl_encoder, f)
