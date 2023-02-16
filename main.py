import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
from Features_engineering import *
from Model import LRlasso as linreg
from sklearn.linear_model import Lasso , LinearRegression

#границы для ограничения выборки
target_bottom = 0 # нижняя граница диапазона целевого параметра
target_top = 1000 # верхняя граница диапазона
status = None     # используется для запуска кода по модели, после введния датасета

st.title ('РАСЧЕТ КОЭФФИЦИЕНТОВ ЛИНЕЙНОЙ МОДЕЛИ')

st.header('1.Ввод данных')


# загрузка датасета
st.markdown('### 1.1 Загрузка данных')
uploaded_file = st.file_uploader("Выберите файл с данными для загрузки")
data = None
if uploaded_file is not None:    
    try:
        data = pd.read_csv(uploaded_file, index_col = 'DateTime')
        st.write(data.head())
    except:
        data = pd.read_csv(uploaded_file)
        st.write("Загрузите данные")  

# выбор что мы хотим предсказывать
st.markdown('### 1.2 Выбор, что хотим предсказать')
if data is not None:
    target = st.selectbox(
        'Выберите целевой параметр для предсказания моделью',
        list(data))
else:
    target = st.text_input('Целевой параметр для разработки модели')
st.write('Введен целевой параметр ', target)

# выбор диапазона целевого параметра
st.markdown('### 1.3 Ввод данных по границе целевого параметра')
if data is not None:    
    try:
        target_bottom = st.number_input('Введите значение ,соответствующее *нижней* границе диапазона целевой переменной')
        st.write('Нижняя граница ', target_bottom)
        target_top = st.number_input('Введите значение ,соответствующее *верхней* границе диапазона целевой переменной')
        st.write('Верхняя граница ', target_top)
    except:
        st.write(" ")


# выбор параметра по которому бедт ориентироваться функция сигмойды - разбиваться датасет
st.markdown('### 1.4 Выбор параметра для разбивки датасета')
if data is not None:
    boundary = st.selectbox(
        'Выберите ключевой параметр для построения логики модели',
        list(data))
else:
    boundary = st.text_input('Ключевой параметр для построения логики модели')
st.write('Введен ключевой параметр ', boundary)

features = [boundary]

# выбор параметров для создания уравнения
st.markdown('### 1.5 Выбор списка параметров -признаков модели')
if data is not None:
    features_ = st.multiselect(
        'Выберите параметры которые будут использованы для построения модели',
        list(data))
    features= features+features_
    st.write('Вы выбрали:', features)
else:
    st.write("Пока данные не загружены. вы не можете выбирать признаки модели")


if data is not None: # это большое УСЛОВИЕ, чтобы не появлялись ошибки при выводе
    try:
        data = DataEngineering(data, features, target).cured_data()
        data = data[data[target] > target_bottom]
        data = data[data[target] < target_top]
    except:
        data = DataEngineering(data, features, target).cured_data()

if st.button('Начать расчет'):
    st.write('Производиться расчет')
    status = True
else:
    st.write('Жду команды для выполнения расчета')

    #########################################################################################

    # ПРОГРАММКА

if status == True:
    # подбираем лучшее значение разбиения 

    list_r2 = [] # это список для записи рассчитанных значений
    df_list_1 = []  # это список для датасетов коэффиицентов левой модели
    df_list_2 = []  # это список для датасетов коэффиицентов правая модели

    intercept_list_1 = [] # это список для значения пересечения оси 0Y левой модели
    intercept_list_2 = [] # это список для значения пересечения оси 0Y правой модели

    try:
        x_range = np.linspace(data[boundary].quantile(q = 0.10),data[boundary].quantile(q = 0.90), 3 )
    except:
        st.write('Пока еще не ввели ключевой тэг для логики')


    for value in x_range:
        _data = DataEngineering(data, features, target).rolling_features() # это копия данных с добавленим скользящего среднего
        _data = _data.merge(data[target], how = 'left', left_index=True, right_index=True)
        _data.dropna(inplace = True)
        data1 = _data[_data[boundary] >= value] # данные СПРАВА от границы разделения
        data2 = _data[_data[boundary] < value] # данные СЛЕВА от границы разделения

        
        # учим модель слева
        data_, best_alpha, best_shot = DataEngineering(data1, features, target).get_new_dataframe(method = 'rfe')
        model1 = linreg(data_,best_shot, target)
        x_train, x_test, y_train_, y_test_ = model1.data_split()
        model1 = Lasso(alpha=0.1).fit(x_train, y_train_)
        #prediction1 = model1.predict(data_[best_shot])
        _data['prediction1'] = model1.predict(_data[best_shot])
        #prediction1 = pd.DataFrame(prediction1, index = data_.index, columns = ['prediction'])

        coef_list1 = list(model1.coef_) 
        dict_coef1 = {    'coef' : np.round(coef_list1, 3),
                   'avg_value' : list(np.round(_data[best_shot].mean().values, 1)),
              'stand_dev_value' : list(np.round(_data[best_shot].std().values, 1)),
                }
        df_coef1 = pd.DataFrame(dict_coef1, index=best_shot)
        intercept_value1 = np.round(model1.intercept_, 3)


        
        
        # учим модель справа
        data_, best_alpha, best_shot = DataEngineering(data2, features, target).get_new_dataframe(method = 'rfe')
        model2 = linreg(data_,best_shot, target)
        x_train, x_test, y_train_, y_test_ = model2.data_split()
        model2 = Lasso(alpha=0.1).fit(x_train, y_train_)
        _data['prediction2'] = model2.predict(_data[best_shot])
        #prediction2 = model1.predict(data_[best_shot])
        #prediction2 = pd.DataFrame(prediction2, index = data_.index, columns = ['prediction'])
        coef_list2 = list(model2.coef_) 
        dict_coef2 = {    'coef' : np.round(coef_list2, 3),
                   'avg_value' : list(np.round(_data[best_shot].mean().values, 1)),
              'stand_dev_value' : list(np.round(_data[best_shot].std().values, 1)),
                }
        df_coef2 = pd.DataFrame(dict_coef2, index=best_shot)
        intercept_value2 = np.round(model2.intercept_, 3)
    

        # используем функцию сигмойды для генерации объединениного предсказания
        # в общем виде функция:  prediction = sigmoida * prediction1 + (1-sigmoida) * prediction2
        # в общем виде функция sigmoida:  sigmoida = (1+e^(-x))^(-1)

        # соединяем все предсказания
        _data['sigmoida'] = 1/(1+np.exp(-1*(_data[boundary] - value)))  
        _data['prediction'] = (1-_data['sigmoida'])*_data['prediction1']+(_data['sigmoida'])*_data['prediction2']
        
        # расчитываем метрики
        r2 = r2_score(_data[target], _data['prediction'])
        list_r2.append(r2)
        df_list_1.append(df_coef1)
        df_list_2.append(df_coef2)

        intercept_list_1.append(intercept_value1)
        intercept_list_2.append(intercept_value2)




    #st.write('Список всех r2', list_r2)
    max_r2 = np.max(list_r2)
    dict_r2 = dict(zip(list_r2, x_range))
    best_boundary_value = dict_r2[max_r2] # это то значение разбивки при котором получается лучшее r2

    # получаем 
    dict_df1 = dict(zip(list_r2, df_list_1))
    best_df1 = dict_df1[max_r2] 
    dict_intercept1 = dict(zip(list_r2, intercept_list_1))
    best_intercept1 = dict_intercept1[max_r2]

    dict_df2 = dict(zip(list_r2, df_list_2))
    best_df2 = dict_df2[max_r2] 
    dict_intercept2 = dict(zip(list_r2, intercept_list_2))
    best_intercept2 = dict_intercept2[max_r2]

    # для отрисовка графика по распределению
    title_text = "Распределение ключевого параметра {} для поcтроения логики".format(boundary)
    fig, ax = plt.subplots()
    ax.hist(data[boundary], bins=20, color = 'grey', alpha = 0.75)
    ax.axvline(x = best_boundary_value, c = 'red', ls= '--')
    ax.set_title(title_text)


    st.header('2. Промежуточный результат (подбор разбиения выборки)')
    #st.write('Лучшее значение метрики R2', np.round(max_r2, 2))
    st.write('Лучшее значение границы для ключевого параметра деления датасета', np.round(best_boundary_value,1))
    st.write(' ')
    st.pyplot(fig)
    st.write(' ')

    # 3 часть выводы результатов
    # создаем таблицу коэффициентов для первой модели (справа) от границы
    #data1 = data[data[boundary] >= best_boundary_value] # данные СПРАВА от границы разделения
    #data_, best_alpha, best_shot = DataEngineering(data1, features, target).get_new_dataframe(method = 'rfe')
    #model1 = linreg(data_,best_shot, target)
    #model1, best_params1, df_coef1, intercept_value1 = model1.model_train(alpha = best_alpha)

    # создаем таблицу коэффициентов для второй модели (слева) от границы
    #data2 = data[data[boundary] < best_boundary_value] # данные СлЕВА от границы разделения
    #data_, best_alpha, best_shot = DataEngineering(data2, features, target).get_new_dataframe(method = 'rfe')
    #model2 = linreg(data_,best_shot, target)
    #model2, best_params2, df_coef2, intercept_value2 = model2.model_train(alpha = best_alpha)



    # создаем таблицу коэффициентов для третьей модели просто регрессия
    data3 = data.copy() # все данные
    data_, best_alpha, best_shot3 = DataEngineering(data3, features, target).get_new_dataframe(method = 'rfe')
    x_train, x_test, y_train_, y_test_ = train_test_split(data_[best_shot3], data_[target],test_size=0.25, random_state=17)
    model3 = LinearRegression().fit(x_train, y_train_)
    r2_model3_test = r2_score(model3.predict(x_test), y_test_)
    r2_model3 = r2_score(model3.predict(data_[best_shot3]), data_[target])
    mae_model3 = mean_absolute_error(model3.predict(data_[best_shot3]), data_[target])

    coef_list3 = list(model3.coef_) 
    dict_coef3 = {    'coef' : np.round(coef_list3, 3),
                   'avg_value' : list(np.round(data_[best_shot3].mean().values, 1)),
              'stand_dev_value' : list(np.round(data_[best_shot3].std().values, 1)),
                }
    df_coef3 = pd.DataFrame(dict_coef3, index=best_shot3)
    intercept_value3 = np.round(model3.intercept_, 3)


    #выводы на экран
    st.header('3. Результаты расчета')
    if r2_model3 > max_r2:
        st.markdown('### 3.1 Результаты без разбиения на данных')
        st.write("Коэффициенты модели")
        st.table(df_coef3)
        st.write("**Пересечение графика вертикальной осью в точке**", np.round(intercept_value3, 1))
        st.write(" ")
        st.write("*Метрика модели R2 =  *", np.round(r2_model3, 2))
        st.write(" ")
        st.write("*Метрика модели MAE =  *", np.round(mae_model3, 2))

        # для отрисовка графика по предсказаням модели
        st.markdown('### 3.2 Фактическое значение целевого параметра и расчетной значение по модели') 
        fig, ax = plt.subplots(figsize=(25, 15))
        ax.plot(model3.predict(data_[best_shot3]), label = 'prediction', color = 'black', alpha = 0.75)
        ax.plot(data_[target], label = 'true', color = 'red', ls = '--')
        ax.set_xticklabels(labels = data_.index,  rotation = 90, fontsize=3)    
        st.pyplot(fig)
        st.write(' ')
    else:    
        st.markdown('### 3.1 Модель для данных выше ключевого значения')

        st.write(" ")
        st.write("Метрика модели R2 =  ", np.round(max_r2, 2))


        st.write("Коэффициенты модели")
        st.table(best_df1)
        st.write("**Пересечение графика вертикальной осью в точке**", best_intercept1)

        st.markdown('### 3.2 Модель для данных ниже ключевого значения ')
        st.write("Коэффициенты модели")
        st.table(best_df2)
        st.write("**Пересечение графика вертикальной осью в точке**", best_intercept2)

        st.markdown('### 3.3 Ключевое значение')
        st.write("Название ключевого параметра ", boundary)
        st.write("**Значение ключевого параметра**", np.round(best_boundary_value,1))

        #st.markdown('### 3.4 Фактическое значение целевого параметра и расчетной значение по модели')
        #st.line_chart(prediction[['prediction', target]])


        



else: # конец большого УСЛОВИЯ, чтобы не появлялись ошибки при выводе
    st.write("*******************************************************")
    #st.header("2. Пока данные не введены, раздел по расчету модели не выполняется")

