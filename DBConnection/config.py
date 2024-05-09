from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS

chatbot = Flask(__name__)
chatbot.config['SQLALCHEMY_DATABASE_URI'] = '***REMOVED***'
chatbot.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(chatbot)
CORS(chatbot)