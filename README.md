# PREDICCIÓN DE TASAS DE CAMBIO USANDO ALGORTIMOS DE MACHINE LEARNING

Descripción General

Este proyecto implementa una interfaz de usuario interactiva para entrenar modelos de Machine Learning en la predicción de tasas de cambio entre dos divisas. A través de la interfaz, los usuarios pueden seleccionar diferentes modelos de predicción, entrenar con datos históricos y visualizar tanto los resultados actuales como las predicciones futuras de manera gráfica y textual.

Características

Modelos Disponibles:

LSTM: Redes neuronales recurrentes para series temporales.
Híbrido: Combinación de convoluciones y LSTM.
Regresión Lineal: Modelo estadístico simple.
Bosques Aleatorios: Método de ensemble para predicción.
Entradas del Usuario:

Primera divisa (por ejemplo, USD).
Segunda divisa (por ejemplo, COP).
Número de años de datos históricos para entrenamiento.
Selección del modelo deseado.
Salidas:

Métrica de pérdida del modelo (MSE).
Gráfico de predicción actual.
Predicciones futuras en formato textual.
Gráfico de las predicciones futuras.

Python 3.9 o superior
Librerías necesarias:
tensorflow
keras
numpy
pandas
scikit-learn
statsmodels
matplotlib
seaborn
yfinance
gradio
Pillow

Estructura del Proyecto

App.py: Archivo principal que contiene la implementación de la interfaz y las funciones de entrenamiento.

FUNCIONES.py: Archivo con las funciones auxiliares utilizadas para el procesamiento, entrenamiento y predicción.

Modelos Incluidos:
LSTM
Híbrido (Conv-LSTM)
Regresión Lineal
Bosques Aleatorios

Introduce los siguientes datos:

Divisa 1: Código de la divisa base (ejemplo: USD).
Divisa 2: Código de la divisa destino (ejemplo: COP).
Número de años: Duración del periodo histórico para entrenamiento.
Modelo: Selecciona uno de los modelos disponibles.

Haz clic en el botón "Entrenar y Predecir".

Explora los resultados:
Métrica de pérdida (MSE).
Gráfico de predicciones sobre el conjunto de prueba.
Predicciones futuras en texto. (Próximos dos datos)
Gráfico de las predicciones futuras.
