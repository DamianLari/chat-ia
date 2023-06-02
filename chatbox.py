import numpy as np
#from tensorflow.keras.models import Sequential
import json
import random
from tensorflow import keras
from keras.models import Sequential, load_model
from keras.layers import Input, Embedding, LSTM, Dense, Bidirectional, Concatenate
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Lire le fichier intents.json
with open('intents.json', 'r') as file:
    data = json.load(file)



dialogues = []
tags = []

# Tokenizer les dialogues
tokenizer = Tokenizer()
tokenizer.fit_on_texts(dialogues)
total_words = len(tokenizer.word_index) + 1

all_words = []
for intent in data['intents']:
    for pattern in intent['patterns']:
        words = pattern.split()
        all_words.extend(words)

# Entraîner le tokenizer sur tous les mots possibles
tokenizer.fit_on_texts(all_words)

# Convertir les phrases en séquences
for intent in data['intents']:
    for pattern in intent['patterns']:
        # Assurez-vous que le modèle est une chaîne
        if isinstance(pattern, str):
            token_list = tokenizer.texts_to_sequences([pattern])[0]
            dialogues.append(token_list)
            tags.append(intent['tag'])


print(dialogues)
# Créer des séquences d'entrée
input_sequences = []
for line in dialogues:
    for i in range(1, len(line)):
        n_gram_sequence = line[:i+1]
        input_sequences.append(n_gram_sequence)

# Padding des séquences
max_sequence_len = max([len(x) for x in dialogues])
if max_sequence_len == 0:
    raise ValueError("Max sequence length is 0. Please check your input data.")
input_sequences = np.array(pad_sequences(dialogues, maxlen=max_sequence_len, padding='pre'))

# Après avoir ajouté tous les mots au tokenizer, calculer total_words
total_words = len(tokenizer.word_index) + 1

# Encoder les tags
tag2index = {tag: index for index, tag in enumerate(set(tags))}
index2tag = {index: tag for tag, index in tag2index.items()}
tags = np.array([tag2index[tag] for tag in tags])

# Créer des prédicteurs et des labels
#predictors, label = input_sequences[:,:-1], tags
predictors, label = input_sequences, np.array(tags)

# Créer le modèle
model = Sequential()
model.add(Embedding(total_words, 10, input_length=max_sequence_len))
model.add(LSTM(150))
model.add(Dense(len(set(tags)), activation='softmax'))  # Le nombre de neurones de sortie correspond au nombre de tags uniques
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entraîner le modèle
model.fit(predictors, label, epochs=100)

# Discuter avec le chatbot
while True:
    print("\nYou: ", end="")
    user_input = input()

    # Prédire l'intention de l'entrée de l'utilisateur
    user_input = ' '.join(input().split())  # Joindre les mots en une seule chaîne
    token_list = tokenizer.texts_to_sequences([user_input])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted_intent = index2tag[np.argmax(model.predict(token_list))]


    # Sélectionner une réponse aléatoire de l'intention prédite
    for i in data['intents']:
        if i['tag'] == predicted_intent:
            print("Chatbot: " + random.choice(i['responses']))
            break