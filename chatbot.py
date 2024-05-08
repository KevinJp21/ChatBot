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
CORS(chatbot)
chatbot.config['SQLALCHEMY_DATABASE_URI'] = '***REMOVED***'
chatbot.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(chatbot)

class User(db.Model):
    __tablename__ = 'usuarios'
    ID_Usu = db.Column(db.Integer, primary_key=True)
    Nombre = db.Column(db.String(255))
    Apellido = db.Column(db.String(255))

class Cita(db.Model):
    __tablename__ = 'citas'
    ID_Cita = db.Column(db.Integer, primary_key=True)
    ID_Paciente= db.Column(db.Integer, db.ForeignKey('usuarios.ID_Usu'))
    FechaCita = db.Column(db.DateTime)
    Motivo = db.Column(db.String(255))
    Estado = db.Column(db.String(50))
    
lemmatizer = WordNetLemmatizer()
model = load_model('DocMe.h5')
with open('intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)
with open('words.pkl', 'rb') as file:
    words = pickle.load(file)
with open('classes.pkl', 'rb') as file:
    classes = pickle.load(file)

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

def predict_class(sentence, threshold=0.383):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    if np.max(res) < threshold:
        return None
    max_index = np.where(res == np.max(res))[0][0]
    category = classes[max_index]
    return category

def handle_greeting(user_id):
    user = User.query.get(user_id)
    if user:
        full_name = f"{user.Nombre} {user.Apellido}"
    else:
        full_name = "Usuario"
    for intent in intents['intents']:
        if intent['tag'] == 'saludo':
            response = random.choice(intent['responses'])
            return response.replace("{name}", full_name)
    return "Hola, ¿en qué puedo ayudarte?"

def handle_last_appointment(user_id):
    ultima_cita = Cita.query.filter_by(ID_Paciente=user_id).order_by(Cita.FechaCita.desc()).first()
    if ultima_cita:
        fecha_cita = ultima_cita.FechaCita.strftime('%Y-%m-%d')
        for intent in intents['intents']:
            if intent['tag'] == 'ultima_cita':
                response = random.choice(intent['responses'])
                return response.replace("{date}", fecha_cita)
    else:
        return "No tienes citas anteriores registradas."

def get_response(tag, user_id):
    handlers = {
        'saludo': handle_greeting,
        'ultima_cita': handle_last_appointment
        # Puedes agregar más handlers aquí.
    }
    if tag in handlers:
        return handlers[tag](user_id)
    else:
        return "Lo siento, no puedo ayudarte con eso."

@chatbot.route('/message', methods=['POST'])
def get_bot_response():
    user_data = request.json
    sentence = user_data.get('message').lower()
    user_id = user_data.get('user_id', 26)  # Asumimos que user_id viene con el request, sino usamos un default

    tag = predict_class(sentence)
    if tag is None:
        return jsonify({"response": "No te entendí lo que me dijiste, prueba otra vez."})

    response = get_response(tag, user_id)
    return jsonify({"response": response})

if __name__ == "__main__":
    from waitress import serve
    serve(chatbot, host="0.0.0.0", port=8080)