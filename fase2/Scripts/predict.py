import argparse
import pandas as pd
import joblib
import os

# Configurar el parser de argumentos
parser = argparse.ArgumentParser()
parser.add_argument('--input_file', required=True, type=str, help='Input CSV file with test data')
parser.add_argument('--model_file', required=True, type=str, help='Trained model file path')
parser.add_argument('--output_file', required=True, type=str, help='Output CSV file for predictions')

args = parser.parse_args()

# Asignar los valores de los argumentos
input_file = args.input_file
model_file = args.model_file
output_file = args.output_file

# Verificar si el archivo del modelo existe
if not os.path.exists(model_file):
    raise FileNotFoundError(f"El archivo de modelo {model_file} no existe.")

# Cargar el modelo entrenado
model = joblib.load(model_file)

# Cargar datos de entrada
data = pd.read_csv(input_file)

# Imprimir las columnas del DataFrame
print("Columnas en el DataFrame de entrada:", data.columns.tolist())

# Aplicar el mismo preprocesamiento que en train.py
X = data.drop("RENDIMIENTO_GLOBAL", axis=1, errors='ignore')  

# Aplicar One-Hot Encoding a las variables categóricas como en el entrenamiento
X = pd.get_dummies(X, drop_first=True)

# Cargar las características originales usadas para entrenar el modelo
# Necesitamos asegurarnos de que las columnas coincidan con las del entrenamiento
model_columns = model.feature_names_in_

# Añadir columnas faltantes y asegurar que el orden sea el correcto
for col in model_columns:
    if col not in X.columns:
        X[col] = 0  # Añadir cualquier columna faltante con valor 0

X = X[model_columns]  # Reordenar las columnas para que coincidan con las del entrenamiento

# Realizar predicciones
predictions = model.predict(X)

# Guardar las predicciones en un archivo CSV
output = pd.DataFrame(predictions, columns=["Prediction"])
output.to_csv(output_file, index=False)

print(f"Predicciones guardadas en {output_file}.")
