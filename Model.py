
# Импорты нужных библиотек
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import xgboost as XGB
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
#import tensorflow as tf
from Features_engineering import *


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
            
    def model_train(self, alpha = 0.1):    
        """
        Обучаем модель с сеткой параметров"""
                       
        x_train, x_test, y_train_, y_test_ = self.data_split() # это функция написана для общего класса моделей
        lin_reg  = Lasso(alpha = alpha, random_state=17)
        model = GridSearchCV(lin_reg, self.DICT_PARAMETERS).fit(x_train, y_train_)
        best_params = model.best_params_
        model = Lasso(alpha = best_params['alpha'], tol = best_params['tol'],  random_state=17).fit(x_train, y_train_)

        coef_list = list(model.coef_) 
        dict_coef = {    'coef' : np.round(coef_list, 3),
                   'avg_value' : list(np.round(self.data[self.features].mean().values, 1)),
              'stand_dev_value' : list(np.round(self.data[self.features].std().values, 1)),
                }
        df_coef = pd.DataFrame(dict_coef, index=self.features)
        intercept_value = np.round(model.intercept_, 3)

        return model, best_params, df_coef, intercept_value

class LRwithSigmoida(LRlasso):
    '''
     Класс LRwithSimoida для создания  стекинговой модели линейной регрессии, состоящей из двух линейных регрессий объединеных по формуле:
     модель = сигмойда*модель1 + (1- сигмойда)* модель 2.
     Сигмойда рассчитвается с учетом лучшего деления данных - это минимальная логика позволяет уулучшит селективность моедли не теряя качетсва.
     Значение симойды коруглется и равно либо "0" либо "1".

    для иницилизации модели необходимы подготовленные данные - data,
    список признаков по которым будет строить модель - features,
    целевая переменная -  target
    метод rolling_features добавляетя признаки с лагом - скользящее срденее за период от сейчас до указанного значения, с шагмо 1 период
    метод get_best_tag_Lasso определяет список тэгов с лучшими коэффициентам детерминации, длинна списка определяется значением TAGS_AMOUNT
    метод model_train объединяет несколько методов и позвоялет быстро создать модель
    
    '''
    DICT_PARAMETERS =  {'alpha': [0.1, 0.25, 0.5, 0.625, 0.7, 0.9],
                         'tol': [0.0001, 0.001, 0.01, 0.1]}
    TAGS_AMOUNT = 5
    START = 12 # минмиальный период для расчета скользящего среднего для метода rolling_features
    END = 12  # вмакисмиальный период для расчета скользящего среднего для метода rolling_features


    def __init__(self, data, features, target, boundary_name):         
        self.data = data
        self.features = features
        self.target = target
        self.boundary_name = boundary_name
        print('Linear regression with sigmoida ready to learn')
        print(self.DICT_PARAMETERS)

    def model_train(self, alpha = 0.1):    
        """
        Обучаем модель с сеткой параметров"""
                       
        x_train, x_test, y_train_, y_test_ = self.data_split() # это функция написана для общего класса моделей
        lin_reg  = Lasso(alpha = alpha, random_state=17)
        model = GridSearchCV(lin_reg, self.DICT_PARAMETERS).fit(x_train, y_train_)
        print('Best parameters', model.best_params_)
        
        return model

    def left_model(self, boundary_name, boundary_value, tags_amount = 5, image = False):
        '''
        Строиться модель, состоящая из двух линенйых моделей и третье модели объединяющий их через значения сигмойды.
        Cигмойда расчитывается с учетом границы разделяющего параметра boundary_name/boundary_value.
        
        boundary_name - наименованеи параметра по которому разделяются данные,
        boundary_value - значнеие прамтера для разделения данных,
        image =True - потсроение графиков для моделей)
        '''
        #1 выбираем данные только слева от границы ключевого признака
        process_1 = self.data[self.data[boundary_name] < boundary_value]
        X_1 = process_1[self.features]
        
        #2 Добавляем средние значения X за период 12 (12*5 минут = 60 минут)
        data_refinery = DataEngineering(data = self.data, features = self.features, target = self.target)
        data_, best_alpha, best_shot = data_refinery.get_new_dataframe()
        
        #3 Обучае модель
        model_1_ = LRlasso(data_, best_shot, self.target).model_train(alpha = best_alpha)
        best_shot_1 = best_shot

        #4 Получаем коэффициенты модели
        coef_list_1 = list(model_1.coef_) 
        dict_coef = {    'coef' : np.round(coef_list_1, 3),
             'avg_value' : list(np.round(self.data[best_shot_1].mean().values)),
            'stand_dev_value' : list(np.round(self.data[best_shot_1].std().values)),
            }
        df1 = pd.DataFrame(dict_coef, index=coef_list_1)
        intercept_1 = model_1.intercept_

        return model_1_, df1, intercept_1

    def right_model(self, boundary_name, boundary_value, tags_amount = 5, image = False):   
               
        '''
        Строиться модель, состоящая из двух линенйых моделей и третье модели объединяющий их через значения сигмойды.
        Cигмойда расчитывается с учетом границы разделяющего параметра boundary_name/boundary_value.
        

        csv_name - имя файла,
        targer_name - какой параметр будем подбирать,
        boundary_name - наименованеи параметра по которому разделяются данные,
        boundary_value - значнеие прамтера для разделения данных,
        image =True - потсроение графиков для моделей)
        '''
    
        process_2 = self.data[self.data[boundary_name] >= boundary_value]
        X_2 = process_2[self.features]
        
        #2. Добавляем средние значения X за период 12 (12*5 минут = 60 минут)
        X_2 = self.rolling_features(X_2, start = self.START, end = self.END)
        X_2 = X_2.fillna(value = X_2.mean())
        features_2 = list(X_2)
        
        model_2 = LRlasso(process_2, features_2, self.target)
        R2, best_alpha, best_shot_2,  best_lasso = model_2.get_best_tag_Lasso()
        model_2_ = LRlasso(process_2, best_shot_2, self.target)
        model_2_= self.model_train()

        return model_2_, best_shot_2
        
    def union_model(self, boundary_name, boundary_value): 
        d = pd.DataFrame()
        model_1, best_shot_1  = self.left_model(boundary_name, boundary_value)
        model_2, best_shot_2 = self.right_model(boundary_name, boundary_value)

        d['model_1'] = model_1.predict(self.data[best_shot_1])
        d['model_2'] = model_2.predict(self.data[best_shot_2])
        d['sigmoida'] = 1/(1+np.exp(-1*(self.data[boundary_name] - boundary_value)))      
        
        union_model = (1-self.data['sigmoida'])*self.data['model_1']+(self.data['sigmoida'])*self.data['model_2']
        
        return union_model

    def find_best_parameters(self, boundary_name):
        '''поиск лучших параметров для логической функции.
        здесь запускается функции union_model при различном значении для логической функции разбивающая диапазон изменения параметра на два интервала -левый- и -правый-.
        Разбивка интревала осущестляется несколько раз и соответнно рассчитвается несколько раз и приниматеся наилучшее значение.'''
        
        list_r2 = []
        list_mae = []
        df_result = []

        x_range = np.linspace(self.data[boundary_name].quantile(q = 0.25),self.data[boundary_name].quantile(q = 0.75), 5 )
        
        for value in x_range:
            union_model = self.union_model(boundary_name, value)
            r2 = r2_score(self.data[self.target], union_model)
            mae= mean_absolute_error(self.data[self.target], union_model)
            list_r2.append(r2)
            list_mae.append(mae)
        
        max_r2 = np.max(list_r2)
        dict_r2 = dict(zip(list_r2, x_range))
        best_boundary_value = dict_r2[max_r2]
        
        return best_boundary_value

    def get_equation(self):

        """Функция получает коэффициенты для двух регрессий и значение лучшего разбиения для логической функции.
        """
        print('старт')
        print(list(self.data))
        # 1 
        # преобразование данных добавление скользящих средних отболр лучших совпадений
        # делается через класс DataEngineering 
        data_refinery = DataEngineering(self.data, self.features, self.target)
        data_ = data_refinery.rolling_features()
        # у нас датафрейм поменялся и теперь его надо обратно пересобрать , чтобы подавать на функцию лассо
        features_ = list(data_)
        data_ = data_.merge(self.data[self.target], how = 'left', left_index=True, right_index=True)
        data_.dropna(inplace = True)
        # переназанчение данные, чтобы попали новые данные после первой функции
        self.data = data_
        self.features = features_
        print('промежуточный')
        print(list(self.data))
        
        #2
        # находим лучшее значения для разделения ключевого признака для логистической функции
        best_boundary_value = self.find_best_parameters(self.boundary_name) # получаем лучшее значение для функции разделения датасетов

        #3
        #делаем модели
        model_1, best_shot_1 = left_model(self.boundary_name, boundary_value = best_boundary_value, tags_amount = 5, image = False)# обучаем модель слева  
        model_2, best_shot_2 = right_model(self.boundary_name, boundary_value = best_boundary_value, tags_amount = 5, image = False)# обучаем модель слева

        # создаем таблицу коэффициентов для первой модели (слева) от границы
        coef_list_1 = list(model_1.coef_) 
        dict_coef = {    'coef' : np.round(coef_list_1, 3),
             'avg_value' : list(np.round(self.data[best_shot_1].mean().values)),
            'stand_dev_value' : list(np.round(self.data[best_shot_1].std().values)),
            }
        df1 = pd.DataFrame(dict_coef, index=coef_list_1)
        intercept_1 = model_1.intercept_



        # создаем таблицу коэффициентов для первой модели (слева) от границы
        coef_list_2 = list(model_2.coef_) 
        dict_coef = {    'coef' : np.round(coef_list_2, 3),
             'avg_value' : list(np.round(self.data[best_shot_2].mean().values)),
            'stand_dev_value' : list(np.round(self.data[best_shot_2].std().values)),
            }
        df2 = pd.DataFrame(dict_coef, index=coef_list_2)
        intercept_2 = model_2.intercept_




        return df1, intercept_1, df2, intercept_2, best_boundary_value