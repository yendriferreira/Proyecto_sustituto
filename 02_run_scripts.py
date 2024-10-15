# -*- coding: utf-8 -*-
"""02 - run scripts"""

import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Cargar los datos de entrenamiento desde train.csv
train_data = pd.read_csv('fase1/train.csv')

# Separar características (X) y etiquetas (y)
X = train_data.drop("RENDIMIENTO_GLOBAL", axis=1)  # Características
y = train_data["RENDIMIENTO_GLOBAL"]  # Etiqueta

# Convertir las características categóricas en variables dummy (one-hot encoding)
X = pd.get_dummies(X, drop_first=True)

# Dividir los datos en conjunto de entrenamiento y prueba (opcional)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenar el modelo de DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Guardar el modelo entrenado
joblib.dump(model, 'modelo_entrenado.pkl')
print("Modelo entrenado y guardado como 'modelo_entrenado.pkl'.")

# Realizar predicciones en los datos de prueba
y_pred = model.predict(X_test)

# Evaluar la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo en el conjunto de prueba: {accuracy:.3f}")

# Guardar las predicciones en un archivo CSV
predicciones_df = pd.DataFrame(y_pred, columns=["Prediction"])
predicciones_df.to_csv("predicciones.csv", index=False)

# Visualización de resultados (Predicciones vs Valores reales)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.xlabel("Valor Real")
plt.ylabel("Predicción")
plt.title("Predicciones vs Valores Reales")
plt.savefig("predicciones_vs_valores_reales.png")
print("Gráfico guardado como 'predicciones_vs_valores_reales.png'.")
plt.show()