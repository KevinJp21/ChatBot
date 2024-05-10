import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import unicodedata
from nltk.corpus import wordnet
from sklearn.model_selection import train_test_split
from keras.regularizers import l2
from keras.callbacks import EarlyStopping

# Descarga de recursos necesarios de nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Inicializar SnowballStemmer para español
stemmer = SnowballStemmer('spanish')
stop_words = stopwords.words('spanish')
ignore_letters = ['?', '!', '¿', '.', ',']

def preprocess_text(text):
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

def add_typo(word):
    if len(word) > 3:
        typo_index = random.randint(1, len(word) - 2)
        word = word[:typo_index] + word[typo_index + 1] + word[typo_index] + word[typo_index + 2:]
    return word

def use_synonym(sentence):
    words = sentence.split()
    selected_word = random.choice(words)
    syns = wordnet.synsets(selected_word)
    if syns:
        synonym = random.choice(syns).lemmas()[0].name().replace('_', ' ')
        sentence = sentence.replace(selected_word, synonym)
    return sentence

def augment_data(pattern):
    augmented_sentences = [pattern, use_synonym(pattern)]
    if random.random() > 0.5:
        augmented_sentences.append(' '.join(add_typo(word) for word in pattern.split()))
    return augmented_sentences

for intent in intents['intents']:
    for pattern in intent['patterns']:
        augmented_patterns = augment_data(pattern)
        for aug_pattern in augmented_patterns:
            word_list = preprocess_text(aug_pattern)
            words.extend(word_list)
            documents.append((word_list, intent['tag']))
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

words = sorted(set(words))

with open('words.pkl', 'wb') as f:
    pickle.dump(words, f)
with open('classes.pkl', 'wb') as f:
    pickle.dump(classes, f)

training = []
output_empty = [0] * len(classes)
for document in documents:
    bag = [1 if word in document[0] else 0 for word in words]
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])
random.shuffle(training)
training = np.array(training, dtype=object)

train_x, test_x, train_y, test_y = train_test_split(list(training[:,0]), list(training[:,1]), test_size=0.2)

model = Sequential()
model.add(Dense(100, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(50, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.6))
model.add(Dense(len(train_y[0]), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
train_process = model.fit(np.array(train_x), np.array(train_y), epochs=100, batch_size=5, verbose=2, validation_data=(np.array(test_x), np.array(test_y)), callbacks=[early_stop])
model.save("DocMe.h5")
