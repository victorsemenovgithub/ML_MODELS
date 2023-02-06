from Model import Model
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import xgboost as XGB
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from Features_engineering import *

class NeuNet(Model):

    '''
    Класс NeuNet для создания прсотой сети sequantial 
    для иницилизации модели необхоидмы подготовленные данные - data,
    список признаков по которым будет строить модель - features,
    целевая переменная -  target
    '''
    LOSS = tf.keras.losses.MeanAbsoluteError()                  # функция потерь сети
    
    OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=1e-3)    # функция оптимизатора
    
    def __init__(self, data, features, target):         
        self.data = data
        self.features = features
        self.target = target
        print('Neural net ready to learn')
        print('LOSS', self.LOSS)
        print('OPTIMIZER', self.OPTIMIZER)

        
        
    def get_compiled_model(self):
        
        #Функция возвращает скомпилированную модель 
        

        INPUT_DIM = len(self.features)  # количество фитчей - входной размер матрицы

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(256, input_dim=INPUT_DIM, activation='relu'))
        model.add(tf.keras.layers.Dense(256, activation='relu'))
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dense(16, activation='relu'))
        model.add(tf.keras.layers.Dense(8, activation='relu'))
        model.add(tf.keras.layers.Dense(1))
        
        model.compile(loss=self.LOSS, optimizer=self.OPTIMIZER)
    
        return model
        
    def model_train(self, train_image = False):    
        
        #Обучаем модель по выбранной архитектуре сети.
        #Возвращает опционально график обучения
        
        #model = self.get_compiled_model()        
        #x_train, x_test, y_train_, y_test_ = self.data_split() # это функция написана для общего класса моделей
        #history = model.fit(x_train, y_train_, validation_data=(x_test, y_test_), epochs=25, batch_size=32, verbose=1)
        
        #if train_image == True:
        #    history_df = pd.DataFrame(history.history)
        #    plt.plot(history_df["loss"],label='train loss')
        #    plt.plot(history_df["val_loss"], label='test val_loss')
        #    plt.legend()
        #    plt.show()

        return model