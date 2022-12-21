"""
Подготовлены функции для провдения анализа и проверки данных.
Предстваляют собой аналог чек-листа
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn import linear_model

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import scipy.stats as st
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, silhouette_score

from matplotlib import pyplot as plt
import seaborn as sns

def quantity_nan(data, bound = 0.05, tags_list = True, df = True):
    '''
    Функция принимает датафрейм.
    bound - граница отсечки допустимой доли nan в данных
    tags_list - возвращает список тэгов входящих в допустимую границу отсечки.

    Рассчитывает кол0во nan и notnan .
    делается оценка количество тэгов с допустимым кол-вом nan.
    Возвращается датафрейм с расчитанным количеством nan.

    '''
    
    nan_ = data.isna().sum()
    nan_ = pd.DataFrame(nan_, columns = ['nan'])

    notnan_ = data.notna().sum()
    notnan_ = pd.DataFrame(notnan_, columns = ['notnan'])

    if df == True:
        df_nan_counts = nan_.merge(notnan_, left_index=True, right_index=True)
        df_nan_counts['quantity_nan'] = df_nan_counts['nan']/df_nan_counts['notnan']
        df_nan_counts['quantity_nan'] = np.round(df_nan_counts['quantity_nan'], 2)
    
    if tags_list == True:
       quantity_list = list(df_nan_counts[df_nan_counts['quantity_nan'] <= bound].index)
    
    return quantity_list, df_nan_counts

def sticking_tag(data, kurtosis_value = 10, skew_value = 2):
    '''
    Функция возращает лист тэгов без "западаний", например датчик долгое время показывает одно и тоже значение.
    Западания ищуться за счет сравнения с коэфициентами эксцесса (kurtosis) и ассиметрии (skew).
    Нормальное распределение имеет значения коэффициентов равно 0 и 0.

    '''
    
    kurtosis = data.kurtosis()
    skew = data.skew()

    unchaged_tag = []
    for i in list(data):
        skew_ = skew[i]
        kurtosis_ = kurtosis[i]
    
        if np.absolute(kurtosis_) < kurtosis_value:
            if np.absolute(skew_) < skew_value:
                unchaged_tag.append(i)
        else:
            pass

    return unchaged_tag

def wave_calculations(data):
    
    """
    Функция расчета параметров "цикличности" появления значений , например nan.
    Делает расчет частоты появления значения.
    Максимальной продолжительности периода с данным значением.
    Возвращает датфрейм с характеристиками.
    """
    general_long = data.shape[0]
    etalon_start = [str, float]
    etalon_finish = [float, str]
    tag_list = list(data)
    wave_dict = {'tag': ['quantity_nan_period', 'relative_nan_period', 'max_nan_period', 'max_relative_nan_period']}
    
    #выбираем цикличность какого значения опредлеяем, пока только nan, a не int
    #assert type_value == type(type_value), 'Ну пока это значение не nan, ничего не получится'
    # пока какая-то не логичная логика получается
    a = 'a'
    
    for tag in tag_list: # заход в цикл по разным датчикам
        #кто первый
        who_the_first = data[tag].fillna(a)[0]
        s = data[tag].fillna(a)
        start_list = []
        finish_list = []
        
        
        for i in list(range(general_long-1)):  # перебираем значения внутри одной серии
          
            if [type(s[i]), type(s[i+1])] == etalon_start:
                start_list.append(i)
                                                         
            elif  [type(s[i]), type(s[i+1])] == etalon_finish:
                finish_list.append(i)
            else:
                pass
    
     
        if  who_the_first == a:
            length_nan = [0]
            for i in list(range(len(start_list)-1)):
                length = finish_list[i] - start_list[i]
                length_nan.append(length)
        else:
            length_nan = [0]
            for i in list(range(len(start_list)-1)):
                length = start_list[i] - finish_list[i]
                length_nan.append(length)
    
        quantity_nan_period = len(length_nan) # количество участков со значением nan    
        relative_nan_period = quantity_nan_period/data.shape[0] # относительное количество участков со значением nan    
        max_nan_period = np.max(length_nan) # макисмальная длинна участка со значением nan    
        max_relative_nan_period = max_nan_period/data.shape[0] # относительная макисмальная длинна участка со значением nan
            
        dictionary = dict.fromkeys([tag], [quantity_nan_period, relative_nan_period,max_nan_period, max_relative_nan_period])
        wave_dict.update(dictionary)
        wave_df = pd.DataFrame(wave_dict)
        wave_df.index = wave_df['tag']
        wave_df = wave_df.drop(columns = ['tag'])

    print('quantity_nan_period', ' - количество участков со значением nan ') 
    print('qrelative_nan_period', ' - относительное количество участков со значением nan ') 
    print('max_nan_period', ' - максимальная длинна участка со значением nan ') 
    print('max_relative_nan_period', ' - относительная максимальная длинна участка со значением nan ') 

    return wave_df

def data_filter (data, process_limitations, sticking_list = True):
    import sys
    sys.path.append('C:\\Users\\User\\Documents\\my_functios')
    import check_list_for_data as check

    sticking = sticking_tag(data) # список не залипших тэгов

    # датафрейм циклочности пропусков
    
    if sticking_list == True:
        wave_= wave_calculations(data = data[sticking]) 
    else:
        wave_= wave_calculations(data = data) 
    wave_ = wave_.transpose()
    rolling_window = wave_['max_nan_period']
    
    
    '''
    Первый и второй пункт оставлют только качетвенные тэги, без большого количества пустых данных и запавших значений. Из третьего пункта можно понять, как нам работать с оставшимися пропусками тэгами. Например при сочетании разных значений показателя: - quantity_nan_period количество участков со значением nan - quantity_nan_period количество участков со значением nan

Если макимальная длинна участка меньше периода 1 часа: - мы можем применять классическую интерполяцию для заполнения всех пустот или апроксимацию при работе на реальных данных.

Если максимальная длинна участка болеее периода 1 часа: - целесообразно принять среднее или медиану или скользящее среднне с периодом равным периоду пропуска для заполнения nan при работе на реальных данных. Например, период пропуска 24х15 мин = 6 часов --- скользящее среднее за 6 часов.

Если quantity_nan_period количество участков со значением nan высокое значение более 250 и низкое значение quantity_nan_period количество участков со значением nan, то получается датчик выпадает часто, но не на долго, можно заполнять скользящим средним с периодом равынм quantity_nan_period
    '''


