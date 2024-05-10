import pickle
import unicodedata

import nltk
import numpy as np
from flask import Flask, jsonify, request
from keras.models import load_model
from nltk.stem import SnowballStemmer
from spellchecker import SpellChecker

import Handlers.handlers as hl
from DBConnection.config import chatbot

stemmer = SnowballStemmer('spanish')
model = load_model('DocMe.h5')
spell = SpellChecker(language='es')

# Cargar palabras y clases
with open('words.pkl', 'rb') as file:
    words = pickle.load(file)
with open('classes.pkl', 'rb') as file:
    classes = pickle.load(file)

def correct_spelling(sentence_words):
    corrected_words = [spell.correction(word) if spell.unknown([word]) else word for word in sentence_words]
    return corrected_words

def clean_up_sentence(sentence):
    sentence = unicodedata.normalize('NFC', sentence.lower())
    sentence_words = nltk.word_tokenize(sentence)
    # Corrige la ortografía antes de aplicar el stemming
    sentence_words = correct_spelling(sentence_words)
    sentence_words = [stemmer.stem(word) for word in sentence_words]
    return sentence_words
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        if w in words:
            bag[words.index(w)] = 1
    return np.array(bag)

def predict_class(sentence, threshold=0.25):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    print(f"DEBUG: Sentence: {sentence}")
    print(f"DEBUG: Bag of Words: {bow}")
    print(f"DEBUG: Prediction: {res}")
    if np.max(res) < threshold:
        print("DEBUG: No prediction exceeded the threshold.")
        return None
    max_index = np.where(res == np.max(res))[0][0]
    category = classes[max_index]
    print(f"DEBUG: Predicted category: {category}")
    return category

def get_response(tag, user_id):
    handlers = {
        'saludo': hl.handle_greeting,
        'ultima_cita': hl.handle_last_appointment,
        'informacion_asistente': hl.handle_infoAssist,
        'proxima_cita': hl.handle_next_appointment,
        'agradecimiento': hl.handle_thankfull,
        'datos_privados': hl.handle_privateDatas,
        'estado_animo_mal': hl.handle_moodState
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