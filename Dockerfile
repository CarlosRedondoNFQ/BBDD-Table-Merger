# Indica la imagen base que se utilizará como punto de partida
FROM python:3.13-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia los archivos de requerimientos al contenedor
COPY requirements.txt .

# Instala las dependencias de la aplicación
RUN pip install --no-cache-dir -r requirements.txt



# Descarga el modelo durante el build
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Fuerza modo offline en runtime (opcional pero recomendable)
ENV HF_HUB_DISABLE_TELEMETRY=1
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1


# Copia el código fuente de la aplicación al contenedor
COPY . .

# Expone el puerto en el que la aplicación escuchará (importante para Cloud Run/GKE)
EXPOSE 8002

# Define la variable de entorno para el entorno de la aplicación (opcional)
ENV APP_ENV=production

# Comando para ejecutar la aplicación cuando el contenedor se inicie
CMD ["streamlit", "run", "main.py", "--server.port", "8002"]