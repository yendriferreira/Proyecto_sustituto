from flask import request, jsonify
from app import app
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'modelo_entrenado.pkl')
DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/datos_entrenamiento.csv')

model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Modelo no entrenado. Use el endpoint /train primero.'}), 400

    data = request.get_json()
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return jsonify({'prediction': prediction.tolist()})

@app.route('/train', methods=['POST'])
def train():
    df = pd.read_csv(DATA_PATH)
    X = df.drop('target', axis=1)
    y = df['target']

    new_model = RandomForestClassifier()
    new_model.fit(X, y)

    joblib.dump(new_model, MODEL_PATH)
    global model
    model = new_model

    return jsonify({'status': 'Modelo entrenado correctamente'})
