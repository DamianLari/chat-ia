data = {"intents": [
             {"tag": "greeting",
              "patterns": ["Hello", "La forme?", "yo", "Salut", "ça roule?"],
              "responses": ["Salut à toi!", "Hello", "Comment vas tu?", "Salutations!", "Enchanté"],
             },
             {"tag": "age",
              "patterns": ["Quel âge as-tu?", "C'est quand ton anniversaire?", "Quand es-tu né?"],
              "responses": ["J'ai 25 ans", "Je suis né en 1996", "Ma date d'anniversaire est le 3 juillet et je suis né en 1996", "03/07/1996"]
             },
             {"tag": "date",
              "patterns": ["Que fais-tu ce week-end?",
"Tu veux qu'on fasse un truc ensemble?", "Quels sont tes plans pour cette semaine"],
              "responses": ["Je suis libre toute la semaine", "Je n'ai rien de prévu", "Je ne suis pas occupé"]
             },
             {"tag": "name",
              "patterns": ["Quel est ton prénom?", "Comment tu t'appelles?", "Qui es-tu?"],
              "responses": ["Mon prénom est Miki", "Je suis Miki", "Miki"]
             },
             {"tag": "goodbye",
              "patterns": [ "bye", "Salut", "see ya", "adios", "cya"],
              "responses": ["C'était sympa de te parler", "à plus tard", "On se reparle très vite!"]
             }
]}


import json
import string
import random 
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer 
import tensorflow as tf 
from keras import Sequential 
from keras.layers import Dense, Dropout
nltk.download_shell()
nltk.download("punkt")
nltk.download("wordnet")

# initialisation de lemmatizer pour obtenir la racine des mots
lemmatizer = WordNetLemmatizer()

# création des listes
words = []
classes = []
doc_X = []
doc_y = []

# parcourir avec une boucle For toutes les intentions
# tokéniser chaque pattern et ajouter les tokens à la liste words, les patterns et
# le tag associé à l'intention sont ajoutés aux listes correspondantes
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern)
        words.extend(tokens)
        doc_X.append(pattern)
        doc_y.append(intent["tag"])
    
    # ajouter le tag aux classes s'il n'est pas déjà là 
    if intent["tag"] not in classes:
        classes.append(intent["tag"])

# lemmatiser tous les mots du vocabulaire et les convertir en minuscule
# si les mots n'apparaissent pas dans la ponctuation
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in string.punctuation]

# trier le vocabulaire et les classes par ordre alphabétique et prendre le
# set pour s'assurer qu'il n'y a pas de doublons
words = sorted(set(words))
classes = sorted(set(classes))

print(words)
print(classes)
print(doc_X)
print(doc_y)


# liste pour les données d'entraînement
training = []
out_empty = [0] * len(classes)

# création du modèle d'ensemble de mots
for idx, doc in enumerate(doc_X):
    bow = []
    text = lemmatizer.lemmatize(doc.lower())
    for word in words:
        bow.append(1) if word in text else bow.append(0)

    # marque l'index de la classe à laquelle le pattern atguel est associé à
    output_row = list(out_empty)
    output_row[classes.index(doc_y[idx])] = 1

    # ajoute le one hot encoded BoW et les classes associées à la liste training
    training.append([bow, output_row])

# mélanger les données et les convertir en array
random.shuffle(training)
training = np.array(training, dtype=object)

# séparer les features et les labels target
train_X = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# définition de quelques paramètres
input_shape = (len(train_X[0]),)
output_shape = len(train_y[0])
epochs = 200

# modèle Deep Learning
model = Sequential()
model.add(Dense(128, input_shape=input_shape, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(output_shape, activation = "softmax"))

adam = tf.keras.optimizers.Adam(learning_rate=0.01, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy"])

print(model.summary())

# entraînement du modèle
model.fit(x=train_X, y=train_y, epochs=200, verbose=1)

def clean_text(text): 
  tokens = nltk.word_tokenize(text)
  tokens = [lemmatizer.lemmatize(word) for word in tokens]
  return tokens

def bag_of_words(text, vocab): 
  tokens = clean_text(text)
  bow = [0] * len(vocab)
  for w in tokens: 
    for idx, word in enumerate(vocab):
      if word == w: 
        bow[idx] = 1
  return np.array(bow)

def pred_class(text, vocab, labels): 
  bow = bag_of_words(text, vocab)
  result = model.predict(np.array([bow]))[0]
  thresh = 0.2
  y_pred = [[idx, res] for idx, res in enumerate(result) if res > thresh]

  y_pred.sort(key=lambda x: x[1], reverse=True)
  return_list = []
  for r in y_pred:
    return_list.append(labels[r[0]])
  return return_list

def get_response(intents_list, intents_json): 
  tag = intents_list[0]
  list_of_intents = intents_json["intents"]
  for i in list_of_intents: 
    if i["tag"] == tag:
      result = random.choice(i["responses"])
      break
  return result

# lancement du chatbot
while True:
    message = input("")
    intents = pred_class(message, words, classes)
    result = get_response(intents, data)
    print(result)