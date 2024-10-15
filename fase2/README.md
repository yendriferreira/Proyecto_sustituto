Antes de esto pretendemos que ya se extrajo el train.rar, en caso de que no, hágalo. 




Pasos para ejecutar -BUILDING- 

NOTA IMPORTANTE: TENER INICIADO EL DOCKER

comandos en este orden: 

cd fase2

docker build -t modelo-predictivo .

cd ..



Ahora aquí estamos de nuevo en el proyecto, así que tenemos 2 formas de ejecutar

Ejecutando ambos scripts:

comandos en este orden: la primer linea crea modelo_entrado.pkl ejecutando el train.py, la segunda linea, al ya tener dicho modelo creado, lo utiliza y crea el predicciones.csv ejecutando el predict.py

docker run -v ${PWD}:/app modelo-predictivo python fase2/Scripts/train.py --data_file fase1/train.csv --model_file modelo_entrenado.pkl --overwrite_model  
docker run -v ${PWD}:/app modelo-predictivo python fase2/Scripts/predict.py --input_file fase1/test.csv --model_file modelo_entrenado.pkl --output_file predicciones.csv




Ahora, si eliminamos ambos archivos creados anteriormente: predicciones.csv y modelo_entrado.pkl y solo ejecutamos el 02_run-scripts.py obtendremos...

Ejecutar script run_scripts:


comandos en orden:
docker run -v ${PWD}:/app modelo-predictivo python 02_run_scripts.py

Esto nos creara ambos archivos y además nos dara el modelo predictivo ya entrenado, tal cual se pidió para esta entrega, todo con el contenedor de docker y ejecutado en el mismo, teniendo este resultado en la consola 

"Modelo entrenado y guardado como 'modelo_entrenado.pkl'.


Precisión del modelo en el conjunto de prueba: 0.344


Gráfico guardado como 'predicciones_vs_valores_reales.png'."
