import argparse
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
import os

# Configurar el parser de argumentos
parser = argparse.ArgumentParser()
parser.add_argument('--data_file', required=True, type=str, help='a csv file with train data')
parser.add_argument('--model_file', required=True, type=str, help='where the trained model will be stored')
parser.add_argument('--overwrite_model', default=False, action='store_true', help='if sets overwrites the model file if it exists')

args = parser.parse_args()

# Asignar los valores de los argumentos
data_file = args.data_file
model_file = args.model_file
overwrite = args.overwrite_model

# Verificar si el archivo de modelo ya existe y la opción de sobreescritura
if os.path.exists(model_file) and not overwrite:
    raise FileExistsError(f"El archivo {model_file} ya existe. Usa --overwrite_model para sobrescribirlo.")

# Cargar datos de entrenamiento
data = pd.read_csv(data_file)

# Proceso de limpieza de datos
# Convertir las fechas (si existen) a timestamp
if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data['datetime'] = data['datetime'].apply(lambda x: x.timestamp())


# Separar características (X) y etiquetas (y)
X = data.drop("RENDIMIENTO_GLOBAL", axis=1)
y = data["RENDIMIENTO_GLOBAL"]

# Opcional: Convertir las características categóricas en variables dummy (one-hot encoding)
X = pd.get_dummies(X, drop_first=True)

# Entrenar el modelo
model = DecisionTreeClassifier()
model.fit(X, y)

# Guardar el modelo entrenado
joblib.dump(model, model_file)
print(f"Modelo guardado exitosamente en {model_file}.")
