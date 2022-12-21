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

def better_se (data, tag):
    """
    It is required import pandas and numpy.
    Функция принимает пандас датафрейм и подбирает такое значение параметра при котором ст.отклонение принимает минимальное значение.
    Данные отсекатются на диапазоне слева до среднего значения.

    Возвращается словарь {датчик:значение} (при минимальном ст.отклонении)
    и возвращается новый датафрейм с "обрезкой" исходных данных по заданному датчику.
    Принципиально используем ОДИН датчик (ключевой для данных).
    При использовании большего кол-ва очень много данных придется отбросить.
   
    """
    keys_ = [] # список с ключами для создания словаря
    value_ = [] # список со значениями для создания словаря
    dictionary = {} # словарь тэгов и значений для достижения нужного ст.отклонения
   
    new_data = pd.DataFrame()
    d = data.copy()    
    for i in list(d):
        iterable_list = list(np.linspace(0, np.mean(d[i]), num = 10))
        se_list = [] # используется внутри цикла расчета ст.отклонения
        for j in iterable_list:
            d_se = d[d[i] > j]
            se = np.std(d_se[i]) # стандартное отклонение
            se_list.append(se)
            se_min = np.min(se_list)
                
        index_min = se_list.index(se_min)
        value = iterable_list[index_min]
        keys_.append(i)
        value_.append(value)
        dictionary = dict(zip(keys_, value_))
    new_data = d[d[tag] > dictionary.get(tag)]

    return  dictionary, new_data
 
def get_positive_skew(data, tag):
    """
    It is required import pandas and numpy.
    Функция принимает пандас датафрейм и подбирает такое значение параметра при котором ассиметрия становиться положительной.
    Положительная ассиметрия означает, что данные вытянуты вправо (хвост по максимальным значениям),
    при отрицательной ассиметрии много данных слева, т.е. туда поподают все остановы.
    Основная задача функции убрать данные остановов и других не типичных режимов.

    Возвращается словарь {датчик:значение} (при максимальной положительной ассиметрии)
    и возвращается новый датафрейм с "обрезкой" исходных данных по заданному датчику.
    Принципиально используем ОДИН датчик (ключевой для данных).
    При использовании большего кол-ва очень много данных придется отбросить.

    """
    keys_ = [] # список с ключами для создания словаря
    value_ = [] # список со значениями для создания словаря
    dictionary = {} # словарь тэгов и значений для достижения нужной ассиметрии
   
    new_data = pd.DataFrame()
    d = data.copy()    
    for i in list(d):
        iterable_list = list(np.linspace(0, np.mean(d[i]), num = 10))
        skew_list = [] # используется внутри цикла расчета ассиметрии
        for j in iterable_list:
            d_skew = d[d[i] > j]
            skew = d_skew[i].skew() # ассиметрия
            skew_list.append(skew)
            skew_max = np.max(skew_list)
                
        index_max = skew_list.index(skew_max)
        value = iterable_list[index_max]
        keys_.append(i)
        value_.append(value)
        dictionary = dict(zip(keys_, value_))
    new_data = d[d[tag] > dictionary.get(tag)]

    return  dictionary, new_data

def rolling_features(data, start = 10, end = 20):
    """
    Функция для расчета и добавления среднего значения (скользящего среднего) за период между start---end по каждому датчику (параметру).
    На выходе получается датасет с добавлеными столбцами, которые соответсвуют усреднненым значениям параметров за выбранный период.
    """
    d = data.copy()
    columns = list(data)
    period_list = range(start, end+1)
    for step in period_list:
        step = step
        for i in columns:
            new = i + '+' + str(step)
            d[new] = d[i].rolling(step).mean()
    
    return  d   


def isinborder (population, sample, tags, confident = 0.95):
    """
    Функция для определения соответствия режима (параметра) границам доверительного интервала совокупности
    population - общие данные по режиму
    sample - данные по режиму на конкретный момент
    tags - список тэгов по которому проверяется режим
    confident - "уверенность", по умолчанию 95%
    """
    import scipy.stats as st
    for i in tags:
        se = np.std(population[i])
        p = population[i].mean()
        borders = st.norm.interval(confident, loc=p, scale = se) 
        mean = sample[i].mean()
        if mean < borders[1] and mean > borders[0] :
            print(i, "в границах доверительного интервала ", "{:.1f}".format(borders[0]),"{:.1f}".format(mean), "{:.1f}".format(borders[1]))
        else:
            print (i, "вне границ доверительного интервала ", "{:.1f}".format(borders[0]),"{:.1f}".format(mean), "{:.1f}".format(borders[1]))


def mean_interval (population, tags, confident = 0.95):
    """
    Функция для определения соответствия режима (параметра) границам доверительного интервала совокупности
    population - общие данные по режиму
    sample - данные по режиму на конкретный момент
    tags - список тэгов по которому проверяется режим
    confident - "уверенность", по умолчанию 95%
    """
    import scipy.stats as st
    for i in tags:
        se = np.std(population[i])
        p = population[i].mean()
        borders = st.norm.interval(confident, loc=p, scale = se) 

        print(i, "левая_граница-среднее-правая_граница ")
        print(i,  "{:.1f}".format(borders[0]),"{:.1f}".format(p), "{:.1f}".format(borders[1]))
        print('\n')

    left = borders[0]
    right = borders[1]    


    return left, right


def are_you_normal(data):
    from scipy import stats
    
    columns = list(data)
    p_valie_list = []
    keys_ = []
    dictionary = {}
    for column in columns:
        k2, p = stats.normaltest(data[column])
        keys_.append(column)
        p_valie_list.append(p)
        
        dictionary = dict(zip(keys_, p_valie_list))
           
    return (dictionary) 

'''def cluster_finder#
    silhouette_list = []
    unique_labels = []
    for i in list(np.linspace(3, 50, 50)):
        clustering = DBSCAN(eps=i, min_samples=3).fit(x)
        score = silhouette_score(x, clustering.fit_predict(x))
        silhouette_list.append(score)
        number_of_labels = np.unique(clustering.labels_).shape
        unique_labels.append(np.unique(number_of_labels))
    plt.plot(silhouette_list, label = 'silhouette')
    plt.legend()
    plt.show()


    plt.plot(unique_labels, label = 'cluster')
    plt.legend()
    plt.show()'''

def get_correlation(data, tag, numbers_tail = 5, numbers_head = 5, image = True):
    from matplotlib import pyplot as plt
    import seaborn as sns
    '''
    функция для расчета корреляции , возвравщает график - тепловую карту и лист лучших тэгов для выбранного параметра
    data - датафрейм,
    tag - тэг для которого будут возращены график и лист корреляций в виде строки,
    numbers_tail = 5 количество тегов с отрицательной корреляцией,
    numbers_head = 6 количество тегов с положительной корреляцией
    '''
    if type(tag) is not str:
        tag = tag[0]
        print('взял только первый элемент из списка  -', tag )
    numbers_head = numbers_head+1
    
    correlation = data.corr()
    corr_df = pd.DataFrame(correlation[tag].sort_values())
    
    list_corr = list(correlation[tag].sort_values().head(numbers_tail).index)
    list_corr = list_corr + list(correlation[tag].sort_values(ascending = False).head(numbers_head)[1:numbers_head].index)
    best_corr = list_corr
    
    if image == True:
        size2 = np.round(corr_df.shape[0]*0.5)
        plt.figure(figsize=(3, size2))
        sns.heatmap(corr_df, annot=True)
    
    return best_corr

def get_best_tag_Lasso(X, y, alpha_start = 0.1, alpha_end=0.5, alpha_step = 0.1, tags_amount = 10, image  = False):
    
    '''
    Функция принимает датафрейм по X и по y,
    alpha_start , alpha_end, alpha_step, tags_amount
    alpha_start = 0.1, alpha_end=0.5, alpha_step = 0.1 значения для листа-цикла для коэф.регулярицзации
    tags_amount = 10 - кол-во тэгов с наибольщим значениям кофэфифиуентов в Лассо с наибольшим R2

    Возвращает значения
    R2 - наибольший R2 полученный при расчете,
    best_alpha - лучшая alpha,
    best_shot - список "лучщих" тэгов,
    best_lasso - датафрейм с ненулевыми коэффициентами Лассо
    '''
    
        
    #sc = StandardScaler()
    #X_scale = sc.fit_transform(X)
    #y_scale = sc.fit_transform(y)
    #y_scale = np.aaray(y_scale).reshape(-1, 1)
    tags_list = list(X)

    x_train, x_test, y_train_, y_test_ = train_test_split(X, y,
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
    best_shot = list(best_lasso.iloc[:, 0:tags_amount])
  
    if image == True:
        plt.plot(alpha_list,r2_list)
        plt.text(np.min(alpha_list), np.max(r2_list), R2)
        plt.show()
    
    return R2, best_alpha, best_shot,  best_lasso


def quality_nan(data, bound = 0.05, tags_list = True, df = True):
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
        df_nan_counts['quality_nan'] = df_nan_counts['nan']/df_nan_counts['notnan']
        df_nan_counts['quality_nan'] = np.round(df_nan_counts['quality_nan'], 2)
    
    if tags_list == True:
        quality_list = list(df_nan_counts[df_nan_counts['quality_nan'] <= bound].index)
    
    return quality_list, df_nan_counts

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


def time_mashine(data, column = 'index'):
    """
    Функция для преобразования времени из текущего формата в формат год/месяц/день (обычно пераое число месяца).
    Использую для анализа данных для группирвоки по месяцам.
    data  - датафрейм
    column - стольбец с датой для преобразования
    
    на выходе получается датафрейм со столбцом Year_month,
    а также возвращает лист except_list,
    в который записывается порядковый номер строки с непреобразованной датой"""
    
    
    if column == 'index':  
        data['timestamp'] = data.index.astype('datetime64[ns]')
    else:
        data['timestamp'] = data[column].astype('datetime64[ns]')
        

    # раскладываем дату на составляющие
    data['year'] = data["timestamp"].dt.year
    data['month'] = data["timestamp"].dt.month

    number = data.shape[0]
    list_time = []
    # собираем дату обратно

    data = data.reset_index()
    #data.drop('index', axis = 1, inplace=True)

    except_list = []
    for i in list(range(number)):      
        try:
            a = pd.Timestamp(year=int(data['year'][i]), month=int(data['month'][i]), day=1)
        except:
            a = pd.Timestamp(year=int(2000), month=int(1), day=1)
            except_list.append(i)
            
        list_time.append(a)

    data['Year_month'] = list_time # теперь дата только год и месяц
    
    return data, except_list