# PREDICCIÓN DE TASAS DE CAMBIO USANDO ALGORTIMOS DE MACHINE LEARNING

Descripción General

Este proyecto implementa una interfaz de usuario interactiva para entrenar modelos de Machine Learning en la predicción de tasas de cambio entre dos divisas. A través de la interfaz, los usuarios pueden seleccionar diferentes modelos de predicción, entrenar con datos históricos y visualizar tanto los resultados actuales como las predicciones futuras de manera gráfica y textual.

Paquetes requeridos para correr el código principal:

Python 3.9 (Recomendado) o superior

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

App.py: Archivo principal que contiene la implementación de la interfaz en gradio y las funciones de entrenamiento del modelo.
FUNCIONES.py: Archivo con las funciones auxiliares utilizadas para el procesamiento, entrenamiento y predicción.
PROYECTO_GRADO.ipynb: Es el Jupyter Notebook usado para el desarrollo del proyecto

Modelos Incluidos:

LSTM: Redes neuronales recurrentes para series temporales.
Híbrido: Combinación de convoluciones y LSTM.
Regresión Lineal: Modelo estadístico simple.
Bosques Aleatorios: Método de ensemble para predicción.
Entradas del Usuario:

Mecánica de uso de gradio y Hugging Face

Introduce los siguientes datos:

Primera divisa (por ejemplo, USD).
Segunda divisa (por ejemplo, COP).
Número de años de datos históricos para entrenamiento (Mínimo 2).
Selección del modelo deseado.

Haz clic en el botón "Entrenar y Predecir".

Métrica de pérdida del modelo (MSE).
Gráfico de predicción actual.
Predicciones futuras en formato textual (Dos por defecto).
Gráfico de las predicciones futuras junto a los últimos valores reales (10% del tamaño de los datos usados para el entrenamiento).

