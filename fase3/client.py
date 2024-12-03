import requests

# URL base del servidor
BASE_URL = "http://localhost:5000"

def train():
    # Envía una solicitud POST al endpoint /train
    response = requests.post(f"{BASE_URL}/train")
    return response.json()  # Devuelve la respuesta en formato JSON

def predict(data):
    # Envía una solicitud POST al endpoint /predict con datos JSON
    response = requests.post(f"{BASE_URL}/predict", json=data)
    return response.json()  # Devuelve la respuesta en formato JSON

if __name__ == '__main__':
    print("Entrenando el modelo...")
    print(train())  # Llama a train y muestra el resultado

    print("Realizando una predicción...")
    data = {"feature1": 1.5, "feature2": 3.2, "feature3": 0.8}  # Datos de entrada
    print(predict(data))  # Llama a predict con los datos y muestra el resultado
