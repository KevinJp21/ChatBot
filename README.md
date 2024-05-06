# Chatbot con Flask y Procesamiento de Lenguaje Natural

Este proyecto es un chatbot inteligente construido utilizando Flask, una librería de aprendizaje profundo (Keras) y técnicas de Procesamiento de Lenguaje Natural (NLP) con la librería NLTK. El chatbot está diseñado para interactuar en español, proporcionando respuestas automáticas basadas en la intención detectada en las consultas del usuario.

## Características Principales

- **Flask como Framework Backend**: Uso de Flask para crear un servidor web que maneje las solicitudes HTTP y proporcione una API para la interacción con el chatbot.
- **Interacción basada en Intenciones**: El sistema clasifica las entradas de los usuarios según las intenciones predefinidas en un archivo JSON.
- **Modelo de Red Neuronal**: Uso de una red neuronal creada con Keras para clasificar las intenciones a partir del texto de entrada.
- **Procesamiento de Texto con NLTK**: Aplicación de tokenización y lematización para preparar los textos antes de alimentarlos al modelo de clasificación.
- **Respuestas Dinámicas**: Generación de respuestas basadas en la intención detectada, personalizadas con el nombre del usuario.

## Tecnologías Utilizadas

- Python
- Flask
- SQLAlchemy para la gestión de bases de datos
- Keras para el modelado de redes neuronales
- NLTK para el procesamiento de lenguaje natural
- MySQL para la base de datos (configurada en AlwaysData)

## Cómo Iniciar

Para ejecutar este proyecto localmente, sigue estos pasos:

1. **Clona el Repositorio:**
   ```bash
   git clone https://github.com/KevinJp21/ChatBot.git

2. Instala las Dependencias:
   ```bash
   pip install -r requirements.txt

3. Configura la Base de Datos:
Asegúrate de configurar correctamente tus credenciales de MySQL en el archivo de configuración de Flask.

4. Entrena el Modelo:
Si es necesario, reentrena el modelo de red neuronal utilizando el script proporcionado
 ```bash
py training.py.
 ```
5. Inicia el Servidor:
```bash
py chatbot.py

