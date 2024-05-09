from flask import Flask, request, jsonify
from sqlalchemy.sql import text
import json
import random
import pickle
import numpy as np
from keras.models import load_model
import nltk
from nltk.stem import WordNetLemmatizer
from DBConnection.config import db, chatbot

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

def predict_class(sentence, threshold=0.2):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    if np.max(res) < threshold:
        return None
    max_index = np.where(res == np.max(res))[0][0]
    category = classes[max_index]
    return category

def handle_greeting(user_id):
    sql = text("SELECT Nombre, Apellido FROM usuarios WHERE ID_Usu = :user_id")
    result = db.session.execute(sql, {'user_id': user_id})
    user = result.fetchone()
    if user:
        full_name = f"{user.Nombre} {user.Apellido}"
    else:
        full_name = ""
    for intent in intents['intents']:
        if intent['tag'] == 'saludo':
            response = random.choice(intent['responses'])
            return response.replace("{name}", full_name)
    return "Hola, ¿en qué puedo ayudarte?"

def handle_next_appointment(user_id):
    sql = text("SELECT c.FechaCita FROM citas c WHERE c.ID_Paciente = 26 AND c.FechaCita > CURRENT_DATE ORDER BY c.FechaCita ASC LIMIT 1")
    result = db.session.execute(sql, {'user_id': user_id})
    proxima_cita = result.fetchone()
    if proxima_cita:
        fecha_cita = proxima_cita.FechaCita.strftime('%Y-%m-%d')
        hora_cita = proxima_cita.FechaCita.strftime('%H:%M:%S')
        # Asumiendo que tienes un intent para manejar la próxima cita
        for intent in intents['intents']:
            if intent['tag'] == 'proxima_cita':
                response = random.choice(intent['responses'])
                response = response.replace("{date}", fecha_cita)
                response = response.replace("{time}", hora_cita)
                return response
    else:
        return "No tienes próximas citas reservadas."

def handle_last_appointment(user_id):
    # Consulta SQL que obtiene la última cita antes de la fecha actual.
    sql = text("SELECT c.FechaCita FROM citas c WHERE c.ID_Paciente = :user_id AND c.FechaCita < CURRENT_DATE ORDER BY c.FechaCita DESC LIMIT 1")
    result = db.session.execute(sql, {'user_id': user_id})
    ultima_cita = result.fetchone()
    if ultima_cita:
        # Extrae la fecha y la hora de la cita.
        fecha_cita = ultima_cita.FechaCita.strftime('%Y-%m-%d')
        hora_cita = ultima_cita.FechaCita.strftime('%H:%M:%S')  # Formato de 24 horas

        # Suponiendo que deseas usar ambos, la fecha y la hora, en la respuesta:
        for intent in intents['intents']:
            if intent['tag'] == 'ultima_cita':
                response = random.choice(intent['responses'])
                response = response.replace("{date}", fecha_cita)
                response = response.replace("{time}", hora_cita)
                return response
    else:
        return "No tienes citas anteriores registradas."
    
def handle_infoAssist():
    for intent in intents['intents']:
        if intent['tag'] == 'informacion_asistente':
            response = random.choice(intent['responses'])
            return response

def get_response(tag, user_id):
    handlers = {
        'saludo': handle_greeting,
        'ultima_cita': handle_last_appointment,
        'informacion_asistente': handle_infoAssist,
        'proxima_cita': handle_next_appointment

    }
    if tag in handlers:
        handler = handlers[tag]
        if handler.__code__.co_argcount == 0:
            return handler()
        elif handler.__code__.co_argcount == 1:
            return handler(user_id)
    else:
        return "Lo siento, no puedo ayudarte con eso."

@chatbot.route('/message', methods=['POST'])
def get_bot_response():
    user_data = request.json
    sentence = user_data.get('message').lower()
    user_id = user_data.get('user_id', 26)

    tag = predict_class(sentence)
    if tag is None:
        return jsonify({"response": "No te entendí lo que me dijiste, prueba otra vez."})

    response = get_response(tag, user_id)
    return jsonify({"response": response})

if __name__ == "__main__":
    from waitress import serve
    serve(chatbot, host="0.0.0.0", port=8080)