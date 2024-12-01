
# Fase 3: API REST

En esta fase se implementa una API REST utilizando Flask que incluye dos funciones principales:

- **`/train`**: Entrena el modelo con datos estándar.
- **`/predict`**: Realiza predicciones con datos nuevos.

## Requisitos

- Python 3.9 o superior.
- Flask y otras dependencias especificadas en `requirements.txt`.

## Archivos Principales

- `app/apirest.py`: Código de la API REST.
- `data/datos_entrenamiento.csv`: Datos estándar para entrenar el modelo.
- `client.py`: Cliente para interactuar con la API.
- `requirements.txt`: Lista de dependencias necesarias.

## Ejecución Local

1. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Iniciar la API**:
   ```bash
   python -m flask run
   ```

   La API estará disponible en `http://localhost:5000`.

## Uso de la API

### 1. Entrenar el Modelo
- Endpoint: `/train`
- Método: `POST`
- Descripción: Entrena el modelo con los datos disponibles.

Ejemplo:
```bash
curl -X POST http://localhost:5000/train
```

### 2. Realizar una Predicción
- Endpoint: `/predict`
- Método: `POST`
- Descripción: Envía datos nuevos y obtiene una predicción.

Datos de ejemplo:
```json
{
  "feature1": 1.5,
  "feature2": 3.2,
  "feature3": 0.8
}
```

Ejemplo:
```bash
curl -X POST -H "Content-Type: application/json" -d '{"feature1":1.5,"feature2":3.2,"feature3":0.8}' http://localhost:5000/predict
```

## Notas
- Los datos de entrenamiento están en `data/datos_entrenamiento.csv`.
- El modelo entrenado se guarda automáticamente en la carpeta `app` como `modelo_entrenado.pkl`.
