# Utilizar una imagen base de Python
FROM python:3.9-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar el archivo requirements.txt para instalar dependencias primero (mejora caché)
COPY requirements.txt /app/

# Instalar las dependencias del proyecto
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el contenido del proyecto al contenedor
COPY . /app

# Exponer el puerto 5000 para la API REST
EXPOSE 5000

# Comando por defecto para ejecutar la API REST de la fase 3
CMD ["python", "fase3/app/apirest.py"]
