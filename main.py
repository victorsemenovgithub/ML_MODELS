import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from Features_engineering import *
from Model import LRlasso as linreg
from sklearn.linear_model import Lasso

st.title ('РАСЧЕТ КОЭФФИЦИЕНТОВ ЛИНЕЙНОЙ МОДЕЛИ')

st.header('1.Ввод данных')


# загрузка датасета
uploaded_file = st.file_uploader("Выберите файл с данными для загрузки")
data = None
if uploaded_file is not None:    
    data = pd.read_csv(uploaded_file, index_col = 'DateTime')
st.write(data.head())

# выбор что мы хотим предсказывать
if data is not None:
    target = st.selectbox(
        'Выберите целевой параметр для предсказания моделью',
        list(data))
else:
    target = st.text_input('Целевой параметр для разработки модели')
st.write('Введен целевой параметр ', target)

# выбор параметра по которому бедт ориентироваться функция сигмойды - разбиваться датасет
if data is not None:
    boundary = st.selectbox(
        'Выберите ключевой параметр для построения логики модели',
        list(data))
else:
    boundary = st.text_input('Ключевой параметр для построения логики модели')
st.write('Введен ключевой параметр ', boundary)


# выбор параметров для создания уравнения
if data is not None:
    features = st.multiselect(
        'Выберите параметры которые будут использованы для построения модели',
        list(data))

    st.write('Вы выбрали:', features)

data = DataEngineering(data, features, target).cured_data()
#########################################################################################

# САМА ПРОГРАММКА


# подбираем лучшее значение разбиения 

list_r2 = [] # это список для записи рассчитанных значений
df_result = []

try:
    x_range = np.linspace(data[boundary].quantile(q = 0.25),data[boundary].quantile(q = 0.75), 5 )
except:
    st.write('Пока еще не ввели ключевой тэг для логики')


for value in x_range:
    data1 = data[data[boundary] >= value] # данные СПРАВА от границы разделения
    data2 = data[data[boundary] < value] # данные СЛЕВА от границы разделения
    
    # учим модель слева
    data_, best_alpha, best_shot = DataEngineering(data1, features, target).get_new_dataframe()
    model1 = linreg(data_,best_shot, target)
    x_train, x_test, y_train_, y_test_ = model1.data_split()
    model1 = Lasso(alpha=0.1).fit(x_train, y_train_)
    prediction1 = model1.predict(data_[best_shot])
    prediction1 = pd.DataFrame(prediction1, index = data_.index, columns = ['prediction'])
    
    
    # учим модель справа
    data_, best_alpha, best_shot = DataEngineering(data2, features, target).get_new_dataframe()
    model2 = linreg(data_,best_shot, target)
    x_train, x_test, y_train_, y_test_ = model2.data_split()
    model2 = Lasso(alpha=0.1).fit(x_train, y_train_)
    prediction2 = model1.predict(data_[best_shot])
    prediction2 = pd.DataFrame(prediction2, index = data_.index, columns = ['prediction'])
   

    # соединяем все предсказания
    prediction = pd.concat([prediction1, prediction2])
    prediction = data.merge(prediction, how = 'right', left_index=True, right_index=True)
    
    # расчитываем метрики
    r2 = r2_score(prediction[target], prediction['prediction'])
    list_r2.append(r2)    

max_r2 = np.max(list_r2)
dict_r2 = dict(zip(list_r2, x_range))
best_boundary_value = dict_r2[max_r2] # это то значение разбивки при котором получается лучшее r2

st.header('2. Промежуточный результат (подбор разбиения выборки)')
st.write('Лучшее значение метрики R2', np.round(max_r2, 2))
st.write('Лучшее значение границы для ключевого параметра деления датасета', np.round(best_boundary_value,1))
st.write(' ')

# создаем таблицу коэффициентов для первой модели (справа) от границы
data1 = data[data[boundary] >= best_boundary_value] # данные СПРАВА от границы разделения
data_, best_alpha, best_shot = DataEngineering(data1, features, target).get_new_dataframe()
model1 = linreg(data_,best_shot, target)
model1, best_params1, df_coef1, intercept_value1 = model1.model_train(alpha = best_alpha)



# создаем таблицу коэффициентов для второй модели (слева) от границы
data2 = data[data[boundary] < best_boundary_value] # данные СлЕВА от границы разделения
data_, best_alpha, best_shot = DataEngineering(data2, features, target).get_new_dataframe()
model2 = linreg(data_,best_shot, target)
model2, best_params2, df_coef2, intercept_value2 = model2.model_train(alpha = best_alpha)



#выводы на экран
st.header('3. Результаты расчета')
st.markdown('### Модель для данных выше ключевого значения')
st.write("Коэффициенты модели")
st.table(df_coef1)
st.write("**Пересечение графика вертикальной осью в точке**", intercept_value1)

st.markdown('### Модель для данных ниже ключевого значения ')
st.write("Коэффициенты модели")
st.table(df_coef2)
st.write("**Пересечение графика вертикальной осью в точке**", intercept_value2)

st.markdown('### Ключевое значение')
st.write("Название ключевого параметра ", boundary)
st.write("**Значение ключевого параметра**", np.round(best_boundary_value,1))
