import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem.snowball import SnowballStemmer  # Cambio aquí
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import unicodedata

# Descarga de recursos necesarios de nltk
nltk.download('punkt')
nltk.download('stopwords')

# Inicializar SnowballStemmer para español
stemmer = SnowballStemmer('spanish')
stop_words = stopwords.words('spanish')
ignore_letters = ['?', '!', '¿', '.', ',']

def preprocess_text(text):
    # Normalizar y limpiar el texto
    text = unicodedata.normalize('NFC', text.lower())
    words = nltk.word_tokenize(text)
    words = [stemmer.stem(word) for word in words if word not in stop_words and word not in ignore_letters]
    return words

# Cargar los intents
with open('intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

words = []
classes = []
documents = []

# Procesar los patrones y las categorías
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Preprocesamiento del texto
        word_list = preprocess_text(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = sorted(set(words))

# Guardar palabras y clases usando pickle
with open('words.pkl', 'wb') as f:
    pickle.dump(words, f)
with open('classes.pkl', 'wb') as f:
    pickle.dump(classes, f)

# Preparar los datos para el entrenamiento
training = []
output_empty = [0] * len(classes)
for document in documents:
    bag = []
    word_patterns = document[0]
    for word in words:
        bag.append(1 if word in word_patterns else 0)
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])
random.shuffle(training)
training = np.array(training, dtype=object)

# Datos para la red neuronal
train_x = list(training[:,0])
train_y = list(training[:,1])

# Construir y entrenar el modelo
model = Sequential()
model.add(Dense(320, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(250, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Configurar el optimizador
sgd = SGD(learning_rate=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Entrenamiento del modelo
train_process = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=10, verbose=2)
model.save("DocMe.h5", train_process)