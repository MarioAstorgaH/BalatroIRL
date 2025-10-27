# Imagen base con Python compatible con MediaPipe
FROM python:3.11-slim

# Crea el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia los archivos locales al contenedor (opcional, si ya tienes scripts)
COPY . /app

# Instala dependencias necesarias
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    && python3 -m pip install --upgrade pip \
    && pip install mediapipe opencv-python \
    && pip install numpy \
    %% pip install tkinter

# Comando por defecto: abrir shell interactivo
CMD ["bash"]