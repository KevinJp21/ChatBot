# Utiliza una imagen base de Python	
FROM python:3.12.3	
	
WORKDIR /	

# Copia el archivo requirements.txt al contenedor	
COPY requirements.txt .	

# Instala las dependencias de la aplicación	
RUN pip install -r requirements.txt	

# Copia todo el contenido del directorio actual al contenedor	
COPY . .	

# Expone el puerto 5000 (o el puerto en el que se ejecute tu aplicación Flask)	
EXPOSE 5000	

# Comando para ejecutar la aplicación Flask	
CMD ["python", "chatbot.py"]