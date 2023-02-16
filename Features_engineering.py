# Импорты нужных библиотек
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, silhouette_score
from sklearn import linear_model
from sklearn.linear_model import Lasso
from sklearn.feature_selection import RFE



class DataEngineering():  
    
    """ Класс для обработки данных.
    Функция 1: Генерации скользящих средних: на базе существующих признаков формируются дополнительные признаки - усрденение за заданный период.
    Функция 2: Подбор лучших признаков. Используется алгоритм лассо для подбора лучших признаков при заданном кол-ве признаков.
    Функция 3: Совмещение первой и второй функции.
    """

    START = 6
    END = 12
    TAGS_AMOUNT = 6
    LIMIT = 3
    
    def __init__(self, data, features, target):         
        self.data = data
        self.features = features
        self.target = target
    

    def cured_data(self):
        """
        Функция берет данные и убирает пропущенные данные. Обычно целевой тэг имеет много пропусков.
        Поэтому мы вытягиваем целевой тэг добавляем на один период вперед/назад и удаляем пропуски.
        Соединяем обратно с общим датафреймом и возвращаем обратно.
        """

        df = pd.DataFrame(self.data[self.target]) # датасет только с целевым тэгом
        df = df.fillna(method = 'ffill', limit = self.LIMIT)
        df.dropna(inplace = True)

        df_ = pd.DataFrame(self.data[self.features]) # датасет только с признаками
        df_ = df_.fillna(method = 'ffill', limit = self.LIMIT*2)
        df_ = df_.fillna(method = 'bfill', limit = self.LIMIT*2)
        df_.dropna(inplace = True)

        df = df.merge(df_, how = 'left', left_index=True, right_index=True)

        return df




    
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

    def get_new_dataframe(self, method = 'lasso'):
        """
        функция получения нового датасета с лучшими признаками, отоборанными по лассо.

        """
        data_ = self.rolling_features()
        # у нас датафрейм поменялся и теперь его надо обратно пересобрать , чтобы подавать на функцию лассо
        features_ = list(data_)
        data_ = data_.merge(self.data[self.target], how = 'left', left_index=True, right_index=True)
        data_.dropna(inplace = True)
        # переназанчение данные, чтобы попали новые данные после первой функции
        self.data = data_
        self.features = features_
        
        if method == 'lasso':            
            R2, best_alpha, best_shot,  best_lasso = self.get_best_tag_Lasso()
            return data_, best_alpha, best_shot
        
        else:
            best_shot = self.get_best_tag_rfe()
            best_alpha = 0.1
            return data_, best_alpha, best_shot


    
    def get_best_tag_rfe(self, number_of_features = 5):
        """
        Ищем лучшие тэги для модели через метод sklearn RFE
        https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html"""

        estimator = Lasso()
        selector = RFE(estimator, n_features_to_select=number_of_features, step=1)
        selector = selector.fit(self.data[self.features], self.data[self.target]) #np.array(self.features).reshape(-1, 1) появилась ошибка про 2D & 1D array
        best_shot = list(selector.get_feature_names_out())  #Mask feature names according to selected features.

        return best_shot


    
    

