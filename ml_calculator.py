# Импорты нужных библиотек
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.patches as patches
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, silhouette_score
import xgboost as XGB
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import DBSCAN
from sklearn import linear_model
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from Features_engineering import *
from Model import LRlasso as linreg

# заглушка
data = data
features = features 
target = 'target'
boundary = 'boundary'

# подбираем лучшее значение разбиения 

list_r2 = [] # это список для записи рассчитанных значений
df_result = []

x_range = np.linspace(data[boundary].quantile(q = 0.25),data[boundary].quantile(q = 0.75), 5 )

for value in x_range:
    data1 = data[data[boundary] >= value] # данные СПРАВА от границы разделения
    data2 = data[data[boundary] < value] # данные СЛЕВА от границы разделения
    
    # учим модель слева
    data_, best_alpha, best_shot = DataEngineering.get_new_dataframe(data1, features, target)
    model1 = linreg(data_,best_shot, target)
    prediction1 = linreg.get_model()

    # учим модель справа
    data_, best_alpha, best_shot = DataEngineering.get_new_dataframe(data2, features, target)
    model2 = linreg(data_,best_shot, target)
    prediction2 = linreg.get_model()

    # соединяем все предсказания
    prediction = pd.concat([prediction1, prediction2], ignore_index=True)

    # расчитываем метрики
    r2 = r2_score(data[target], prediction)
    list_r2.append(r2)    

max_r2 = np.max(list_r2)
dict_r2 = dict(zip(list_r2, x_range))
best_boundary_value = dict_r2[max_r2] # это то значение разбивки при котором получается лучшее r2




# создаем таблицу коэффициентов для первой модели (справа) от границы
data1 = data[data[boundary] >= best_boundary_value] # данные СПРАВА от границы разделения
data_, best_alpha, best_shot = DataEngineering.get_new_dataframe(data1, features, target)
model1 = linreg(data_,best_shot, target)
model1 = model1.model_train(alpha = best_alpha)

coef_list_1 = list(model_1.coef_) 
dict_coef = {    'coef' : np.round(coef_list_1, 3),
            'avg_value' : list(np.round(data_[best_shot].mean().values)),
        'stand_dev_value' : list(np.round(data_[best_shot].std().values)),
        }
df1 = pd.DataFrame(dict_coef, index=coef_list_1)
intercept_1 = model_1.intercept_

# создаем таблицу коэффициентов для второй модели (слева) от границы
data2 = data[data[boundary] < best_boundary_value] # данные СлЕВА от границы разделения
data_, best_alpha, best_shot = DataEngineering.get_new_dataframe(data2, features, target)
model1 = linreg(data_,best_shot, target)
model1 = model1.model_train(alpha = best_alpha)

coef_list_2 = list(model_2.coef_) 
dict_coef = {    'coef' : np.round(coef_list_2, 3),
            'avg_value' : list(np.round(data_[best_shot].mean().values)),
        'stand_dev_value' : list(np.round(data_[best_shot].std().values)),
        }
df2 = pd.DataFrame(dict_coef, index=coef_list_2)
intercept_2 = model_2.intercept_


