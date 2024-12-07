# %%
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

import statsmodels.graphics.tsaplots as sgt
import matplotlib.pyplot as plt
import tensorflow as tf
import multiprocessing
import seaborn as sns
import yfinance as yf
import pandas as pd
import numpy as np

import warnings
import random
import io

warnings.filterwarnings("ignore")

def computo():# Configura el número de hilos en las operaciones internas
    num_processors = multiprocessing.cpu_count()
    threading.set_intra_op_parallelism_threads(num_processors)
    threading.set_inter_op_parallelism_threads(num_processors)

    print("Using all available processors:", num_processors)

# Agrego semillas aleatorias para asegurarme que los experimentos en este programa sean reproducibles
def randoms(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# %%
#Con esta función le pido al usuario que escriba el par de divisas que quiere evaluar y lo pongo en el formato de Yahoo Finance para la descarga.

def pairs(Curr_1, Curr_2):
    first_currency = Curr_1
    second_currency = Curr_2
    ticker = str(f'{first_currency}{second_currency}=X')
    return ticker

def descargar_información(years, Curr_1, Curr_2):
    #Obtengo los pares de divisas que voy a comparar
    ticker = pairs(Curr_1, Curr_2)

    #Defino los tiempos en los que evaluaré el par de divisas

    end_date = date.today()
    start_date = end_date.replace(year=end_date.year - years)
    end_date, start_date = str(end_date), str(start_date)

    Currencies = yf.download(ticker, start=start_date, end=end_date, progress=False)

    return Currencies, ticker

# %%
def calcular_volatilidad(data):

    # Calcular los rendimientos diarios
    volatilidad = data.pct_change()
    avrg =  np.nanmean(volatilidad)
    volatilidad = np.where(np.isnan(volatilidad), avrg, volatilidad)

    #Ordeno los valores de la volatilidad de menor a mayor
    volatilidad_ord = np.sort(volatilidad)

    #Establezco un alpha del 5% para particionar la cola derecha de la distribución de volatilidad
    alpha = 0.05

    particion_sup = round(len(volatilidad_ord)*(1-alpha/2))
    particion_inf = round(len(volatilidad_ord)*(alpha/2))

    upper_limit = volatilidad_ord[particion_sup]
    lower_limit = volatilidad_ord[particion_inf]


    return lower_limit, upper_limit, volatilidad_ord, volatilidad

def agregar_características_rezagadas(arr=None, num_lags=0, array_outliers=None):
    # Asegúrate de que arr sea un array 1D
    arr = np.asarray(arr).flatten()  # Convierte a un array 1D
    n_samples = len(arr)

    # Crear un nuevo array para almacenar las características de lag
    lagged_data = np.zeros((n_samples - num_lags, num_lags + 1))  # (n_samples - num_lags, num_lags + 1)
    
    # Llenar la primer columna con los datos originales
    lagged_data[:, 0] = arr[num_lags:]  # Los datos originales sin los primeros 'num_lags' elementos

    # Llenar las columnas de lag
    for i in range(1, num_lags + 1):
        lagged_data[:, i] = arr[num_lags - i:n_samples - i]  # Datos con el retraso correspondiente

    vol_array = array_outliers[num_lags:].reshape(-1, 1)
    lagged_data = np.hstack((lagged_data, vol_array))

    return lagged_data

# %%
#La función a continuación sirve para particionar los datos de train, val y para particionar los datos de test. Si el parámetro test = 1, entonces solo transforma los datos de test y no particiona train - val

def particionar_datos(alpha, train_set, test=False):
    if test == False:
        #Hago particiones en mi dataset con un punto de corte del 90%
        train_size = int(len(train_set)*alpha)
       
        # Separo el conjunto de entrenamiento de el de validación
        train_df = train_set[:train_size]
        val_df = train_set[train_size:]

        # Separo X, y en X_train, y_train y X_val, y_val para cada conjunto
        X_train = train_df[:,1:]
        y_train = train_df[:,0]
        X_val = val_df[:,1:]
        y_val = val_df[:,0]

        return X_train, y_train, X_val, y_val
    
    else:
        X_test = train_set[:,1:]
        y_test = train_set[:,0]

        return X_test, y_test

# %%
def plot_series(y=None, y_pred=None, x_label="$t$", y_label="$x(t)$", legend=True):
    plt.figure(figsize=(12,5))
    sns.set(style="whitegrid")

    # Calcular los límites de volatilidad para y
    if y is not None:
        y = y.flatten() if len(y.shape) > 1 else y
        lower_limit_y, upper_limit_y, _, _ = calcular_volatilidad(pd.Series(y))
        outliers_y = [i for i, val in enumerate(y) if val < lower_limit_y or val > upper_limit_y]

    # Graficar los valores reales en línea discontinua y color slategrey
    if y is not None:
        plt.plot(np.arange(len(y)), y, linestyle='--', label="$y(t)$ Real", color='#97a6c4', linewidth=3)

    # Graficar los valores predichos como puntos en color slateblue
    if y_pred is not None:
        plt.plot(np.arange(len(y_pred)), y_pred, linestyle='-.', linewidth=3 ,color='#384860', label="Predicción")

    # Graficar los outliers de y en color crimson
    if y is not None and outliers_y:
        plt.scatter(outliers_y, [y[i] for i in outliers_y], color='#298c8c', s=11, label="Outliers", zorder=5)

    # Configurar etiquetas y estilos
    plt.xlabel(x_label, fontsize=16)
    plt.ylabel(y_label, fontsize=16, rotation=0)
    plt.grid(True)
    plt.autoscale()

    # Mostrar la leyenda solo si hay valores reales o predicciones
    if legend and (y is not None or y_pred is not None):
        plt.legend(fontsize=14, loc="upper left")

    # Guardar el gráfico en un buffer de memoria (BytesIO)
    buf = io.BytesIO()  # Crear un buffer de memoria
    plt.savefig(buf, format='png')  # Guardar la figura como PNG en el buffer
    buf.seek(0)  # Reiniciar el puntero del buffer
    
    return buf
    
    return buf

# %%
def ventana_tiempo(caracteristicas, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(caracteristicas)):
        # Aseguramos que el tamaño de la ventana no salga del rango de los datos
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        
        if out_end_ix > len(caracteristicas):
            break
        
        # Extracto la característica actual como X (de la fila) y la salida como y (de la misma fila)
        seq_x = caracteristicas[i:end_ix, 1:]  # Las 6 características del vector, sin incluir la primera
        seq_y = caracteristicas[i, 0]  # El primer valor de la misma fila
        
        X.append(seq_x)
        y.append(seq_y)
    
    return np.array(X), np.array(y)

n_steps_in, n_steps_out = 5,1 # Número de filas con características que se ingresan al modelo y número de predicciones
num_lags = 5

def plot_learning_curves(loss, val_loss):
    epochs = np.arange(len(loss)) + 1
    sns.set(style="whitegrid")
    
    # Crear la gráfica con Seaborn
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=epochs, y=loss, marker="o", label="Pérdida en entrenamiento", color="slateblue")
    sns.lineplot(x=epochs, y=val_loss, marker="o", label="Pérdida en validación", color="slategrey")
    
    # establezco los límites del eje y para visualizar mejor
    max_value = max(max(loss), max(val_loss))
    plt.ylim(0, max_value)
    
    # etiquetas y legendas de los ejes
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Función de pérdida", fontsize=14)
    plt.title("Curva de aprendizaje", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    
    plt.show()

# %%
class LSTM_Model:
    def __init__(self, n_steps_in=n_steps_in, num_lags=num_lags, seed=42, lr_scheduler=None):
        self.n_steps_in = n_steps_in
        self.num_lags = num_lags
        self.model = self.build_model()
        randoms(seed)

        # Definir un lr_scheduler por defecto si no se pasa uno
        if lr_scheduler is None:
            self.lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
        else:
            self.lr_scheduler = lr_scheduler        

    def build_model(self):
        # Definir el modelo LSTM
        model = Sequential([
            Bidirectional(LSTM(units=256, activation='relu')), 
            Dense(1)  # Capa de salida
        ])
        
        # Compilar el modelo
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def fit(self, X, y, epochs=20, batch_size=60, **kwargs):
        return self.model.fit(X, y, epochs=epochs, batch_size=batch_size, **kwargs, callbacks=[self.lr_scheduler])

    def predict(self, X, **kwargs):
        return self.model.predict(X, **kwargs)
    
    
    def __call__(self):
        # Devolver el modelo directamente cuando se llame a la instancia
        return self.model
    
    def __str__(self):
        # Este método define cómo se representará el nombre de la instancia
        return "LSTM_Model"

# %%
class Conv_LSTM:
    def __init__(self, n_steps_in=n_steps_in, num_lags=0, seed=42, lr_scheduler=None):
        self.n_steps_in = n_steps_in
        self.num_lags = num_lags
        self.model = self.build_model()
        randoms(seed)

        # Definir un lr_scheduler por defecto si no se pasa uno
        if lr_scheduler is None:
            self.lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
        else:
            self.lr_scheduler = lr_scheduler

    def build_model(self):
        # Defino la arquitectura del modelo
        model = Sequential([
            Conv1D(
                filters=256,
                kernel_size=1,
                activation='relu',
                padding='causal'
            ),
            Bidirectional(LSTM(units=256, activation='relu')), 
            Dense(1)  # Capa de salida
        ])

        # Compilar el modelo
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def fit(self, X, y, epochs=20, batch_size=60, **kwargs):
        return self.model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=[self.lr_scheduler], **kwargs)
    
    def predict(self, X, **kwargs):
        return self.model.predict(X, **kwargs)

    def __call__(self):
        # Retorna el modelo directamente
        return self.model
    
    def __str__(self):
        # Este método define cómo se representará el nombre de la instancia
        return "CONV_LSTM_Model"

# %%
#Agrego una clase para ayudar a inicializar el pipeline en la función lag_features
class LagFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, num_lags=None, array_outliers=None):
        self.num_lags = num_lags
        self.array_outliers = array_outliers    

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Llama a la función agregar_características_rezagadas
        return agregar_características_rezagadas(X, self.num_lags, self.array_outliers)
    
def modelos(num_lags):
    lstm = LSTM_Model()
    hybrid = Conv_LSTM()
    lr = LinearRegression()
    rf = RandomForestRegressor()

    return lstm, hybrid, lr, rf

# %%
def entrenar_modelo(alpha, modelo, years=None, Curr_1=None, Curr_2=None, num_lags=None, plot=True):

    num_lags_models = num_lags + 1

    #Inicializo el modelo para que reinicie el entrenamiento desde cero y no desde el último ciclo de entrenamiento
    if isinstance(modelo, LSTM_Model):
        modelo = LSTM_Model(n_steps_in, num_lags_models)  # Reinicia el modelo LSTM
    elif isinstance(modelo, Conv_LSTM):
        modelo = Conv_LSTM(n_steps_in, num_lags_models)  # Reinicia el modelo Conv_LSTM

    # Descargamos los datos y obtenemos el ticker
    data, ticker = descargar_información(years, Curr_1, Curr_2)
    data = data['Close']
    if plot==True:
        print("\n")
        print(f'El par de divisas evaluado es: {ticker}', "\n")

    randoms(42)

    #Calculo la volatilidad de mis datos
    upper_outliers, lower_outliers, volatilidad_datos, volatility_array = calcular_volatilidad(data)

    # Crear el pipeline con el transformador de características de rezago
    pipeline = Pipeline([
        ('scaling', StandardScaler()),
        ('lag_features', LagFeatureTransformer(num_lags=num_lags, array_outliers=volatility_array)),
        ('model', modelo)
    ])

    # Escalamos los datos y agregamos las características de rezago
    scaled_data = pipeline.named_steps['scaling'].fit_transform(data.to_frame())
    data_preprocessed = pipeline.named_steps['lag_features'].fit_transform(scaled_data)

    # Partición inicial de los datos en train y test
    train_size = alpha
    split = int(len(data_preprocessed) * train_size)
    train_df = data_preprocessed[:split]
    test_df = data_preprocessed[split:]

    # Separamos los datos en características (X) y etiquetas (y)
    X_train, y_train, X_val, y_val = particionar_datos(train_size, train_df)
    X_test, y_test = particionar_datos(train_size, test_df, test=1)

    # Si el modelo es LSTM o Conv_LSTM, aplicar ventana de tiempo
    if isinstance(modelo, (LSTM_Model, Conv_LSTM)):
        split_nn = int(len(train_df)*train_size)
        train_df2 = train_df[:split_nn]
        val_df = train_df[split_nn:]

        X_train, y_train = ventana_tiempo(train_df2, n_steps_in, n_steps_out)
        X_val, y_val = ventana_tiempo(val_df, n_steps_in, n_steps_out)
        X_test, y_test = ventana_tiempo(test_df, n_steps_in, n_steps_out)
        pipeline.named_steps['model'].fit(X_train, y_train, verbose=0)
        y_real, y_pred = y_test, pipeline.named_steps['model'].predict(X_test, verbose=0)
    else:
        pipeline.named_steps['model'].fit(X_train, y_train)
        y_real, y_pred = y_test, pipeline.named_steps['model'].predict(X_test)

    y_real = pipeline.named_steps['scaling'].inverse_transform(y_real.reshape(-1, 1))
    y_pred = pipeline.named_steps['scaling'].inverse_transform(y_pred.reshape(-1, 1))
    values = y_real, y_pred

    # Cálculo de la pérdida del modelo
    model_loss = root_mean_squared_error(y_real, y_pred)**2

    if plot==True:
        # Graficar predicciones vs. valores reales
        img_plot = plot_series(y_real, y_pred)
        print(f'La función de pérdida (MSE) del modelo de {modelo} es de: {model_loss}')

    return pipeline, X_test, model_loss, values, img_plot

# %%
def predict_n_steps_ahead(pipeline, initial_data, n_steps=None):
    # Obtener el modelo del pipeline
    model = pipeline.named_steps['model']
    
    # Inicializar last_data con el último registro de initial_data
    last_data = initial_data[-1].copy()  # Último paso de datos (debe ser 1D)

    # Lista para almacenar las predicciones
    rolling_predictions = []
    
    # Realizamos predicciones dependiendo del tipo de modelo
    if isinstance(model, (RandomForestRegressor, LinearRegression)):
        for _ in range(n_steps):
            # Preparamos los datos para la predicción (incluir rendimiento actual)
            input_data = last_data.reshape(1, -1)  # Mantenemos las 6 características
            
            # Realizamos la predicción
            pred = pipeline.named_steps['model'].predict(input_data)  
            rolling_predictions.append(pred[0])  # Almacenamos el valor de predicción
            
            # Actualizamos los datos para la siguiente predicción
            last_data[:-1] = np.roll(last_data[:-1], shift=-1)  # Desplazamos las primeras 5 características hacia la izquierda
            last_data[-2] = pred[0]  # Sustituimos el penúltimo valor con la nueva predicción
            
            # Calcular el rendimiento respecto al paso anterior
            if len(rolling_predictions) > 1:
                rendimiento = (pred[0] - rolling_predictions[-2]) / abs(rolling_predictions[-2])
                last_data[-1] = rendimiento  # Actualizamos el último valor con el nuevo rendimiento
            else:
                last_data[-1] = initial_data[-1,-1]  # Inicializa el rendimiento en el primer paso

        rolling_predictions = np.array(rolling_predictions).reshape(-1, 1)
        rolling_predictions = pipeline.named_steps['scaling'].inverse_transform(rolling_predictions)

    elif isinstance(model, (LSTM_Model, Conv_LSTM)):
        for _ in range(n_steps):
            # Preparamos los datos para la predicción (debe tener forma (1, 5, 6))
            input_data = last_data.reshape(1, 5, -1)
            
            # Realizamos la predicción
            pred = model.predict(input_data)
            rolling_predictions.append(pred[0, 0])  # Almacenamos la predicción
            
            # Actualizamos last_data para la siguiente iteración
            last_data = np.roll(last_data, shift=-1, axis=0)  # Desplazamos los datos hacia arriba
            last_data[-1, :-1] = pred[0, 0]  # Sustituimos la última fila con la predicción
            # Calcular el rendimiento respecto al paso anterior
            if len(rolling_predictions) > 1:
                rendimiento = (pred[0, 0] - rolling_predictions[-2]) / abs(rolling_predictions[-2])
                last_data[-1, -1] = rendimiento  # Actualizamos el rendimiento
            else:
                last_data[-1, -1] = initial_data[-1, -1, -1]  # Inicializa el rendimiento con el último valor original

        rolling_predictions = np.array(rolling_predictions).reshape(-1, 1)
        rolling_predictions = pipeline.named_steps['scaling'].inverse_transform(rolling_predictions)

    return rolling_predictions, last_data
# %%
def imprimir_predicciones(values, rolling_predictions, n_steps=None):# Concatenar los valores
    y_real, y_pred_extract = values
    combined_predictions = np.concatenate([y_pred_extract, rolling_predictions])
    x_values = np.arange(len(combined_predictions))

    # Crear la gráfica
    plt.figure(figsize=(12,5))
    plt.plot(x_values, combined_predictions, marker='o')

    # Crear la gráfica
    plt.plot(x_values[:len(y_pred_extract)], y_pred_extract, label='Valores reales', marker='o', color='#97a6c4')  # Valores actuales en azul
    plt.plot(x_values[len(y_pred_extract):], rolling_predictions, label=f'Predicciones a futuro en {n_steps} paso(s)', marker='o', color='#384860')  # Predicciones futuras en rojo

    plt.xlabel('$(t)$')
    plt.ylabel('$y(t)$')
    plt.legend()
    plt.grid(True)
    plt.title('Valores reales junto a predicciones Futuras')
    plt.show()

    # Guardar el gráfico en un buffer de memoria (BytesIO)
    buf2 = io.BytesIO()  # Crear un buffer de memoria
    plt.savefig(buf2, format='png')  # Guardar la figura como PNG en el buffer
    buf2.seek(0)  # Reiniciar el puntero del buffer
    
    return buf2
# %%
