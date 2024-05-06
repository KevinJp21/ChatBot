from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import json
import random
import pickle
import numpy as np
from keras.models import load_model
import nltk
from nltk.stem import WordNetLemmatizer

chatbot = Flask(__name__)
chatbot.config['SQLALCHEMY_DATABASE_URI'] = '***REMOVED***'
chatbot.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(chatbot)

#BD

class User(db.Model):
    __tablename__ = 'usuarios'
    ID_Usu = db.Column(db.Integer, primary_key=True)
    Nombre = db.Column(db.String(255))
    Apellido = db.Column(db.String(255))

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

def predict_class(sentence, threshold=0.383):  # Ajusta el umbral según sea necesario
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    # Verificar si el máximo valor supera el umbral
    if np.max(res) < threshold:
        return None  # Ninguna predicción es lo suficientemente confiable
    max_index = np.where(res == np.max(res))[0][0]
    category = classes[max_index]
    return category

def get_response(tag, user_name):
    for intent in intents['intents']:
        if intent['tag'] == tag:
            # Elegir una respuesta al azar y formatearla con el nombre del usuario
            response = random.choice(intent['responses'])
            return response.replace("{name}", user_name)
        

# Endpoint de la API
@chatbot.route('/message', methods=['POST'])
def get_bot_response():
    user = User.query.filter_by(ID_Usu=26).first()  # hardcoded for user ID 26
    if user is None:
        return jsonify({"response": "Usuario no encontrado para ID 26."})
    
    full_name = f"{user.Nombre} {user.Apellido}"

    user_data = request.json
    sentence = user_data.get('message')
    tag = predict_class(sentence)
    if tag is None:
        return jsonify({"response": "No te entendí lo que me dijiste, prueba otra vez."})
    response = get_response(tag, full_name)
    return jsonify({"response": response})

if __name__ == "__main__":
    from waitress import serve
    serve(chatbot, host="0.0.0.0", port=8080)