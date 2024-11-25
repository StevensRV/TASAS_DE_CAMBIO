from tensorflow.keras.layers import Conv1D, Input, Bidirectional, MaxPooling1D
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from tensorflow.keras.layers import LSTM, Dense, Flatten, Dropout, Reshape
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.config import threading
from sklearn.pipeline import Pipeline
from keras.models import Sequential
from functools import partial
from datetime import date
from PIL import Image

import statsmodels.graphics.tsaplots as sgt
import matplotlib.pyplot as plt
import tensorflow as tf
import multiprocessing
import seaborn as sns
import yfinance as yf
import pandas as pd
import numpy as np
import gradio as gr
import warnings
import random
import base64
import io

warnings.filterwarnings("ignore")
from FUNCIONES import *

lstm, hybrid, lr, rf = modelos()

# Función para simular el entrenamiento del modelo
def entrenar_y_ejecutar(divisa1, divisa2, anos, modelo):
    if not divisa1 or not divisa2 or not anos or not modelo:
        return "Por favor, completa todos los campos."
    try:
        anos = int(anos)  # Validar que los años sean un número entero
        
        num_lags= 5
        n_steps = 2

        # Selección del modelo según el nombre pasado desde el Dropdown
        if modelo == "LSTM":
            selected_model = lstm
        elif modelo == "Híbrido":
            selected_model = hybrid
        elif modelo == "Regresión Lineal":
            selected_model = lr
        elif modelo == "Bosques Aleatorios":
            selected_model = rf
        else:
            return "Modelo no reconocido."

        pipeline, X_test, model_loss, values, img_plot = entrenar_modelo(alpha=0.9, modelo=selected_model, Curr_1=divisa1,
                                                                          Curr_2=divisa2, years=anos, num_lags=num_lags,
                                                                            plot=True)

        resultado_texto = f'Modelo entrenado con éxito. La pérdida (MSE) del modelo fue: {model_loss:.4f}'
        rolling_predictions, last_data = predict_n_steps_ahead(pipeline, X_test, n_steps)
        preds = imprimir_predicciones(values, rolling_predictions, n_steps=n_steps)

        # Resultado textual y gráfico
        img = Image.open(img_plot)
        img_plot = np.array(img)
        pred = Image.open(preds)
        preds_plot = np.array(pred)

        return resultado_texto, img_plot, rolling_predictions, preds_plot  # Retorna el gráfico y diccionario

    except ValueError:
        return "El número de años debe ser un entero."
    
# Crear interfaz con Gradio
with gr.Blocks() as interfaz:
    gr.Markdown("### PREDICCIÓN DE LA TASA DE CAMBIO USANDO ALGORITMOS DE MACHINE LEARNING")
    
    # Entrada para la primera divisa
    divisa1_input = gr.Textbox(label="Divisa 1 (ejemplo: USD)")
    
    # Entrada para la segunda divisa
    divisa2_input = gr.Textbox(label="Divisa 2 (ejemplo: COP)")
    
    # Entrada para el número de años
    anos_input = gr.Number(label="Número de años", value=1, precision=0)
    
    # Menú desplegable para seleccionar el modelo
    modelo_selector = gr.Dropdown(choices=["LSTM", "Híbrido", "Regresión Lineal", "Bosques Aleatorios"], label="Modelo")
    
    # Botón para ejecutar
    boton_ejecutar = gr.Button("Entrenar y Predecir")
    
    # Área de salida para mostrar el resultado del entrenamiento
    resultado_output = gr.Textbox(label="Resultado")
    
    # Área de salida para mostrar la imagen del gráfico de predicción
    imagen_output = gr.Image(label="Gráfico de Predicción Actual", type='numpy')

    # Nueva área de salida para mostrar las predicciones futuras
    predicciones_output = gr.Textbox(label="Predicciones Futuras", lines=5)

    # Nueva área de salida para mostrar el gráfico de las predicciones futuras
    predicciones_plot_output = gr.Image(label="Gráfico de Predicciones Futuras", type='numpy')

    # Conexión de los inputs con la función
    boton_ejecutar.click(
        entrenar_y_ejecutar,
        inputs=[divisa1_input, divisa2_input, anos_input, modelo_selector],
        outputs=[resultado_output, imagen_output, predicciones_output, predicciones_plot_output]
    )

# Iniciar la interfaz
interfaz.launch()
