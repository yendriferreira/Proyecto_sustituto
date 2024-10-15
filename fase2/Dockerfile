# Utilizar una imagen base de Python
FROM python:3.9-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar los archivos necesarios al contenedor
COPY . /app

# Instalar las dependencias del modelo desde un archivo requirements.txt
RUN pip install pandas joblib scikit-learn matplotlib --no-cache-dir -r requirements.txt


# Copiar todo el contenido del proyecto al contenedor
COPY . .

# Comando por defecto (puede ser sobreescrito)
CMD ["python", "train.py"]
