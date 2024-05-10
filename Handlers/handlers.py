from sqlalchemy.sql import text
from DBConnection.config import db
import random
import json

with open('intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

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
    sql = text("SELECT c.FechaCita, c.Motivo, CONCAT(u.Nombre, ' ', u.Apellido) AS NombreCompleto FROM citas c JOIN usuarios u ON c.ID_Med = u.ID_Usu WHERE c.ID_Paciente = :user_id AND c.FechaCita > CURRENT_DATE ORDER BY c.FechaCita ASC LIMIT 1;")
    result = db.session.execute(sql, {'user_id': user_id})
    proxima_cita = result.fetchone()
    if proxima_cita:
        fecha_cita = proxima_cita.FechaCita.strftime('%Y-%m-%d')
        hora_cita = proxima_cita.FechaCita.strftime('%H:%M:%S')
        doctorName = proxima_cita.NombreCompleto
        reason = proxima_cita.Motivo
        for intent in intents['intents']:
            if intent['tag'] == 'proxima_cita':
                response = random.choice(intent['responses'])
                response = response.replace("{date}", fecha_cita)
                response = response.replace("{time}", hora_cita)
                response = response.replace("{doctor}", doctorName)
                response = response.replace("{reason}", reason)
                return response
    else:
        return "No tienes próximas citas reservadas."

def handle_last_appointment(user_id):
    sql = text("SELECT c.FechaCita, c.Motivo, CONCAT(u.Nombre, ' ', u.Apellido) AS NombreCompleto FROM citas c JOIN usuarios u ON c.ID_Med = u.ID_Usu WHERE c.ID_Paciente = :user_id AND c.FechaCita < CURRENT_DATE ORDER BY c.FechaCita DESC LIMIT 1;")
    result = db.session.execute(sql, {'user_id': user_id})
    ultima_cita = result.fetchone()
    if ultima_cita:
        fecha_cita = ultima_cita.FechaCita.strftime('%Y-%m-%d')
        hora_cita = ultima_cita.FechaCita.strftime('%H:%M:%S')
        doctorName = ultima_cita.NombreCompleto
        reason = ultima_cita.Motivo
        for intent in intents['intents']:
            if intent['tag'] == 'ultima_cita':
                response = random.choice(intent['responses'])
                response = response.replace("{date}", fecha_cita)
                response = response.replace("{time}", hora_cita)
                response = response.replace("{doctor}", doctorName)
                response = response.replace("{reason}", reason)
                return response
    else:
        return "No tienes citas anteriores registradas."

def handle_infoAssist():
    for intent in intents['intents']:
        if intent['tag'] == 'informacion_asistente':
            response = random.choice(intent['responses'])
            return response
        
def handle_thankfull():
    for intent in intents['intents']:
        if intent['tag'] == 'agradecimiento':
            response = random.choice(intent['responses'])
            return response
        
def handle_BadMoodState():
    for intent in intents['intents']:
        if intent['tag'] == 'estado_animo_mal':
            response = random.choice(intent['responses'])
            return response
        
def handle_privateDatas():
    for intent in intents['intents']:
        if intent['tag'] == 'datos_privados':
            response = random.choice(intent['responses'])
            return response
        
def handle_MedicationRecommended():
    for intent in intents['intents']:
        if intent['tag'] == 'recomendacion_medicamento':
            response = random.choice(intent['responses'])
            return response