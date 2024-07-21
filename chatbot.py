import tensorflow as tf
import pickle
import unicodedata
import nltk
import numpy as np
from flask import jsonify, request
from keras.models import load_model
from nltk.stem import SnowballStemmer
from spellchecker import SpellChecker
import Handlers.handlers as hl
from DBConnection.config import chatbot


stemmer = SnowballStemmer('spanish')
model = load_model('DocMe.keras')
spell = SpellChecker(language='es')

# Cargar palabras y clases
with open('words.pkl', 'rb') as file:
    words = pickle.load(file)
with open('classes.pkl', 'rb') as file:
    classes = pickle.load(file)

def correct_spelling(sentence_words):
    # Asegura que la función siempre retorne una lista, incluso si está vacía
    corrected_words = [spell.correction(word) if spell.unknown([word]) else word for word in sentence_words if word is not None]
    return corrected_words

def clean_up_sentence(sentence):
    if not sentence:
        return []  # Retorna una lista vacía si la entrada es None o vacía
    sentence = unicodedata.normalize('NFC', sentence.lower())
    sentence_words = nltk.word_tokenize(sentence)
    # Corrige la ortografía antes de aplicar el stemming
    sentence_words = correct_spelling(sentence_words)
    # Asegúrate de que todas las palabras son válidas antes de aplicar el stemming
    sentence_words = [stemmer.stem(word) for word in sentence_words if word]
    return sentence_words

def analyze_input_quality(sentence_words):
    unknown_words = [word for word in sentence_words if word not in words]
    unknown_ratio = len(unknown_words) / len(sentence_words) if sentence_words else 0
    return unknown_ratio

def adjust_threshold(base_threshold, unknown_ratio, increase_factor=0.1):
    if unknown_ratio > 0.5:
        return base_threshold + increase_factor
    return base_threshold

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        if w in words:
            bag[words.index(w)] = 1
    return np.array(bag)

def predict_class(sentence, base_threshold=0.5):
    sentence_words = clean_up_sentence(sentence)
    unknown_ratio = analyze_input_quality(sentence_words)
    adjusted_threshold = adjust_threshold(base_threshold, unknown_ratio)

    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    print(f"DEBUG: Sentence: {sentence}")
    print(f"DEBUG: Unknown Words Ratio: {unknown_ratio}")
    print(f"DEBUG: Bag of Words: {bow}")
    print(f"DEBUG: Prediction: {[f'{x:.1f}' for x in res]}")
    print(f"DEBUG: Adjusted Threshold: {adjusted_threshold}")
    if np.max(res) < adjusted_threshold:
        print("DEBUG: No prediction exceeded the threshold.")
        return None
    max_index = np.where(res == np.max(res))[0][0]
    category = classes[max_index]
    print(f"DEBUG: Predicted category: {category}")
    return category

def get_response(tag, user_id):
    return hl.handle_response(tag, user_id)

@chatbot.route('/message', methods=['POST'])
def get_bot_response():
    user_data = request.json
    sentence = user_data.get('message').lower()
    user_id = user_data.get('user_id')

    tag = predict_class(sentence)
    if tag is None:
        return jsonify({"response": "No te entendí lo que me dijiste, prueba otra vez."})

    response = get_response(tag, user_id)
    return jsonify({"response": response})
    
print("|--------ChatBot Iniciado--------|")
if __name__ == "__main__":
    from waitress import serve
    serve(chatbot, host="0.0.0.0", port=8080)

