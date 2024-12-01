import requests

BASE_URL = "http://localhost:5000"

def train():
    response = requests.post(f"{BASE_URL}/train")
    return response.json()

def predict(data):
    response = requests.post(f"{BASE_URL}/predict", json=data)
    return response.json()

if __name__ == '__main__':
    print("Entrenando el modelo...")
    print(train())

    print("Realizando una predicción...")
    data = {"feature1": 1.5, "feature2": 3.2, "feature3": 0.8}
    print(predict(data))
