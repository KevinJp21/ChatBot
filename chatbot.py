from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import random
import pickle
import numpy as np
from keras.models import load_model
import nltk
from nltk.stem import WordNetLemmatizer

chatbot = Flask(__name__)
CORS(chatbot)

# Carga los modelos y los archivos necesarios
lemmatizer = WordNetLemmatizer()
model = load_model('DocMe.h5') 
with open('intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

with open('words.pkl', 'rb') as file:
    words = pickle.load(file)

with open('classes.pkl', 'rb') as file:
    classes = pickle.load(file)

# Funciones auxiliares
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        if w in words:
            bag[words.index(w)] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    max_index = np.where(res == np.max(res))[0][0]
    category = classes[max_index]
    return category

def get_response(tag):
    for i in intents['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])

# Endpoint de la API
@chatbot.route('/message', methods=['POST'])
def get_bot_response():
    sentence = request.json['message']
    tag = predict_class(sentence)
    response = get_response(tag)
    return jsonify({"response": response})

# Punto de entrada principal
if __name__ == "__main__":
    chatbot.run(debug=True)