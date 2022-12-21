
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
import shap

class Model:
    '''
    Класс модель для создания различных моделей
    для иницилизации модели необхоидмы подготовленные данные - data,
    список признаков по которым будет строить модель - features,
    целевая переменная -  target
    '''
    
    
    def __init__(self, data, features, target):         
        self.data = data
        self.features = features
        self.target = target

    def data_split(self, test_size = 0.25):        
        '''Метод делит данные в заданной пропорции.
        Пропорция задается через значение - test_size '''
        
        X = self.data[self.features]
        y = self.data[self.target]
        x_train, x_test, y_train_, y_test_ = train_test_split(X, y, test_size=test_size, random_state=17)      
               
        return x_train, x_test, y_train_, y_test_


    def __model_predict(self, model, x_test:'DataFrame | Series' = None) -> 'DataFrame':
        '''формируются предсказания на тестовой выборке и на полных данных. 
        результата используется внутри для расчета метрик'''
        
        # можно получить расчет, как для тестовой выборки и полного ряда данных
        # так и только для полного ряда данных (иногда это удобно) 
        if type(x_test) == 'DataFrame | Series':
            data_predict_test = model.predict(x_test) # предсказания на тестовой выборке
            data_predict_whole = model.predict(self.data[self.features])  # предсказания на общей выборке
            return data_predict_test, data_predict_whole
        else:
            data_predict_whole = model.predict(self.data[self.features])  # предсказания на общей выборке
            return data_predict_whole
    
    def model_metrics(self, model, x_test, y_test_, print_result = True):
        """ Рассчитываются метрики по проекту.
        на вход подаютьсяЖ
        model - обученная модель,
        x_test, y_test_ - тетсовые выборки, если нужны метрик по тесту,
        print_result = True - распечатка результатов"""

        if type(x_test) == 'DataFrame | Series':
            data_predict_test, data_predict_whole = self.__model_predict(model, x_test) # получаем предсказния из функции класса

            R2_test = 'R2_test = '+str('{:.2f}'.format(r2_score(y_test_.values, data_predict_test)))
            MAE_test = 'MAE_test = ' + str('{:.2f}'.format(mean_absolute_error(y_test_.values, data_predict_test)))

            R2 = 'R2 = '+str('{:.2f}'.format(r2_score(self.data[self.target], data_predict_whole)))
            MAE= 'MAE = ' + str('{:.2f}'.format(mean_absolute_error(self.data[self.target], data_predict_whole)))
            
            if print_result == True:
                print(R2_test)
                print(MAE_test)
                print(R2)
                print(MAE)
        
        else:
            data_predict_whole = self.__model_predict(model, x_test) # получаем предсказния из функции класса

            R2 = 'R2 = '+str('{:.2f}'.format(r2_score(self.data[self.target], data_predict_whole)))
            MAE= 'MAE = ' + str('{:.2f}'.format(mean_absolute_error(self.data[self.target], data_predict_whole)))

            if print_result == True:
                print(R2)
                print(MAE)

        return data_predict_whole

    # для графика
    def model_image(self, model):
        ''' отрисовывает график предсказний и целеовго значения по всей выборке'''
        data_predict_whole = self.__model_predict(model) # получаем предсказания из функции класса

        plt.figure(figsize=(24, 8))
        plt.plot(self.data[self.target], label = 'true', color = 'gray', ls = '--', alpha = 0.75)
        plt.plot(self.data.index, data_predict_whole, label = 'predict', color = 'red')
        plt.legend()
        plt.title('True - predict')
        plt.show()

    def get_model(self):

        """ Функция рассчитывает предсказания, выводит метрики и графики."""
        model = self.model_train()  
        x_train, x_test, y_train_, y_test_ = self.data_split() # это функция написана для общего класса моделей
        data_predict_whole = self.model_metrics(model, x_test, y_test_, print_result = True)
        self.model_image(model)

        return data_predict_whole    





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
        """
        Функция возвращает скомпилированную модель 
        """

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
        """
        Обучаем модель по выбранной архитектуре сети.
        Возвращает опционально график обучения"""
        
        model = self.get_compiled_model()        
        x_train, x_test, y_train_, y_test_ = self.data_split() # это функция написана для общего класса моделей
        history = model.fit(x_train, y_train_, validation_data=(x_test, y_test_), epochs=25, batch_size=32, verbose=1)
        
        if train_image == True:
            history_df = pd.DataFrame(history.history)
            plt.plot(history_df["loss"],label='train loss')
            plt.plot(history_df["val_loss"], label='test val_loss')
            plt.legend()
            plt.show()

        return model


class XGBoosting(Model):
    '''
    Класс XGBoosting для создания XGBoost with Gridsearch 
    для иницилизации модели необходимы подготовленные данные - data,
    список признаков по которым будет строить модель - features,
    целевая переменная -  target
    '''

    DICT_PARAMETERS = {'booster' : ['gbtree', 'dart'],
              'max_depth': np.linspace(3, 8, 6,dtype=int),
             'eta': np.linspace(0.05, 0.8, 5)}

    def __init__(self, data, features, target):         
        self.data = data
        self.features = features
        self.target = target
        print('XGBoost ready to learn')
        print(self.DICT_PARAMETERS)
            
    def model_train(self):    
        """
        Обучаем модель с сеткой параметров"""
                       
        x_train, x_test, y_train_, y_test_ = self.data_split() # это функция написана для общего класса моделей
        XGBoost = XGB.XGBRegressor()
        model = GridSearchCV(XGBoost, self.DICT_PARAMETERS).fit(x_train, y_train_)
        print(model.best_params_)
        print("Score {:.3f}".format(model.best_score_))

        return model
    
class LRlasso(Model):
    '''
    Класс LRlasso для создания модели линейной регрессии 
    для иницилизации модели необходимы подготовленные данные - data,
    список признаков по которым будет строить модель - features,
    целевая переменная -  target
    метод rolling_features добавляетя признаки с лагом - скользящее срденее за период от сейчас до указанного значения, с шагмо 1 период
    метод get_best_tag_Lasso определяет список тэгов с лучшими коэффициентам детерминации, длинна списка определяется значением TAGS_AMOUNT
    метод model_train объединяет несколько методов и позвоялет быстро создать модель
    '''

    DICT_PARAMETERS =  {'alpha': [0.1, 0.25, 0.5, 0.625, 0.7, 0.9],
                         'tol': [0.0001, 0.001, 0.01, 0.1]}
    TAGS_AMOUNT = 10
    START = 3 # минмиальный период для расчета скользящего среднего для метода rolling_features
    END = 12  # вмакисмиальный период для расчета скользящего среднего для метода rolling_features

    def __init__(self, data, features, target):         
        self.data = data
        self.features = features
        self.target = target
        print('Linear regression ready to learn')
        print(self.DICT_PARAMETERS)

    def rolling_features(self):
        """
        Функция для расчета и добавления среднего значения (скользящего среднего) за период между start---end по каждому датчику (параметру).
        На выходе получается датасет с добавлеными столбцами, которые соответсвуют усреднненым значениям параметров за выбранный период.
        """
        d = self.data[self.features].copy()
        columns = self.features
        period_list = range(self.START, self.END+1)
        for step in period_list:
            step = step
            for i in columns:
                new = i + '+' + str(step)
                d[new] = d[i].rolling(step).mean()
        
        return  d       
    
    def get_best_tag_Lasso(self, alpha_start = 0.1, alpha_end=0.5, alpha_step = 0.1, image  = False):
        
        '''
        Функция принимает,
        alpha_start , alpha_end, alpha_step, tags_amount
        alpha_start = 0.1, alpha_end=0.5, alpha_step = 0.1 значения для листа-цикла для коэф.регулярицзации
        TAGS_AMOUNT = 10 - кол-во тэгов с наибольщим значениям кофэфифиуентов в Лассо с наибольшим R2

        Возвращает значения
        R2 - наибольший R2 полученный при расчете,
        best_alpha - лучшая alpha,
        best_shot - список "лучших" тэгов,
        best_lasso - датафрейм с ненулевыми коэффициентами Лассо
        '''
                 
        
        tags_list = self.features

        x_train, x_test, y_train_, y_test_ = train_test_split(self.data[self.features], self.data[self.target],
                                                            test_size=0.25,
                                                            random_state=17)
        r2_list = []
        alpha_list = list(np.arange(alpha_start, alpha_end, alpha_step))
        all_data = pd.DataFrame()
        for number,alpha in enumerate(alpha_list):
            lasso = linear_model.Lasso(alpha=alpha)
            model_lasso_ = lasso.fit(x_train, y_train_)
            r2 = '{:.2f}'.format(r2_score(y_test_, model_lasso_.predict(x_test)))
            r2_list.append(r2)
                      
            coef_list = list(model_lasso_.coef_)
            regres_dict = dict(zip(tags_list, coef_list))
            
            df = pd.DataFrame(regres_dict, index = [alpha])
            df = df[df != 0]
            df = df.dropna(axis =1)
            
            all_data = pd.concat([all_data, df])
            all_data = all_data.dropna(axis='columns')
              
        r2_list = np.array(r2_list, dtype = float)
        r2_max = np.max(r2_list)
        index = list(r2_list).index(r2_max)
        R2 = 'R2_max = '+str('{:.2f}'.format(r2_max))    
        
        best_alpha = alpha_list[index]

        best_lasso = np.abs(all_data)
        best_lasso = best_lasso.sort_values(by = best_alpha, axis = 1, ascending = False)
        best_shot = list(best_lasso.iloc[:, 0:self.TAGS_AMOUNT])
    
        if image == True:
            plt.plot(alpha_list,r2_list)
            plt.text(np.min(alpha_list), np.max(r2_list), R2)
            plt.show()
    
        return R2, best_alpha, best_shot,  best_lasso


            
    def model_train(self):    
        """
        Обучаем модель с сеткой параметров"""
                       
        x_train, x_test, y_train_, y_test_ = self.data_split() # это функция написана для общего класса моделей
        lin_reg  = Lasso(random_state=17)
        model = GridSearchCV(lin_reg, self.DICT_PARAMETERS).fit(x_train, y_train_)
        print('Best parameters', model.best_params_)
        
        return model


    