# Chatbot with Flask and Natural Language Processing

This project is an intelligent chatbot built using Flask, a deep learning library (Keras) and Natural Language Processing (NLP) techniques with the NLTK library. The chatbot is designed to interact in Spanish, providing automatic responses based on the intention detected in the user's queries.

<div style="text-align: center;">
    <img src="https://portfolio-kj.vercel.app/assets/docme-chatbot-LzPG2RAO.webp" alt="proyecto DocMe Chatbot" width="300">
</div>

## Main Features

- **Flask as Backend Framework**: Using Flask to create a web server that handles HTTP requests and provides an API for interaction with the chatbot.
- **Interaction based on Intents**: The system classifies user input according to predefined intents in a JSON file.
- **Neural Network Model**: Use of a neural network created with Keras to classify intentions from the input text.
- **Text Processing with NLTK**: Application of tokenization and lemmatization to prepare texts before feeding them to the classification model.
- **Dynamic Responses**: Generation of responses based on the detected intention, personalized with the user's name.

## Used technology

- Python
- Flask
- Flask-SQLAlchemy for database management
- numpy for math operations
- nltk for natural language processing
- keras for neural network modeling
- tensorflow as backend for keras
- waitress as WSGI server
- pymysql for connection with MySQL
- scikit-learn for machine learning techniques
- pyspellchecker for spell checking

## How to Start

To run this project locally, follow these steps:

1. **Clone the Repository:**
```bash
git clone https://github.com/KevinJp21/ChatBot.git
```
2. **Install the Dependencies:**
```bash
pip install -r requirements.txt
```
3. **Configure the Database:**
Make sure you correctly configure your MySQL credentials in the Flask configuration file.

4. **Train the Model:**
If necessary, retrain the neural network model using the provided script
 ```bash
py training.py.
 ```
5. **Start the Server:**
```bash
py chatbot.py

