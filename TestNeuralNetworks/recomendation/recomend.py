import pandas as pd
import numpy as np
# для работы с разреженными матрицами(очень много нулей)
from scipy.sparse import csr_matrix
# алгоритм k-ближайших соседей
from sklearn.neighbors import NearestNeighbors


        ### импортируем файлы foods.csv и countbuy.csv ###
# и преобразуем их в датафреймы. Исп read_csv
foods = pd.read_csv('recom/foods.csv')
countbuys = pd.read_csv('recom/countbuys.csv')


        ### посмотрим на содержимое файла foods.csv ###
# дополнительно удалим столбец 'genres', он нам не нужен
# параметр axis = 1 говорит, что мы работаем со столбцами,
# inplace = True, что изменения нужно сохранить
# и countbuys.csv здесь также удаляем ненужный столбец 'timestamp'
#foods.drop(['genres'], axis=1, inplace=True)
#print(foods.head(5))
#countbuys.drop(['timestamp'], axis=1, inplace=True)
#print(countbuys.head(5))


        ### создаём матрицу предпочтений ###
# для этого воспользуемся сводной таблицей (pivot table)
# по горизонтали будут блюда, по вертикали - пользователи, значения - кол покупок
user_matrix = countbuys.pivot(index='foodId', columns='userId', values='countbuy')


        ### NaN расшифровывается как Not A Number ###
# и представляет собой наиболее частый
# способ отображения пропущенных значений.
# Так как мы будем заниматься вычислениями расстояния,
# каждое значение таблицы должно быть числовым.
# С помощью функции fillna() мы заменим NaN на ноль
user_matrix.fillna(0, inplace=True)
#print(user_matrix.head())


        ### Теперь давайте уберем неактивных пользователей ###
# и блюда с небольшим количеством оценок
# вначале сгруппируем(объединим) пользователей, возьмем только столбец 'countbuy'
# и посчитаем, сколько было покупок у каждого пользователя
# сделаем то же самое, только для блюд
users_votes = countbuys.groupby('userId')['countbuy'].agg('count')
#print(users_votes)
foods_votes = countbuys.groupby('foodId')['countbuy'].agg('count')
#print(foods_votes)


        ### теперь создадим фильтр (mask) ###
user_mask = users_votes[users_votes > 8].index
food_mask = foods_votes[foods_votes > 1].index
# применим фильтры и отберем блюда с достаточным количеством покупок
# а также активных пользователей
user_matrix = user_matrix.loc[food_mask, :]
user_matrix = user_matrix.loc[:, user_mask]
#теперь давайте посмотрим на новую размерность, то есть на те блюда #
# и тех пользователей, которые остались после фильтрации данных
#print(user_matrix.shape)
#print(user_matrix.head(8))


        ### Если столбцов очень много(а у нас их уже довольно много), ###
# то говорят про данные с высокой размерностью(high-dimensional data).
# В таком формате алгоритм будет долго обсчитывать расстояния между блюдами.
# Для того, чтобы преодолеть эту сложность можно преобразовать
# данные в формат сжатого хранения строкой(сompressed sparse row, csr)
# атрибут values передаст функции csr_matrix только значения датафрейма
csr_data = csr_matrix(user_matrix.values)
#print(csr_data[:8, :8])


### Остается только сбросить индекс для удобства поиска рекомендованных блюд ###
user_matrix = user_matrix.rename_axis(None, axis=1).reset_index()
#print(user_matrix.head(20))


        ### Все, данные готовы для обучения модели ###
# для наших целей нам достаточно измерить расстояние между объектами
# создадим объект класса NearestNeighbors
# metric = ‘cosine’ - выбираем способ измерения расстояния, это будет косинусное сходство
# algorithm = ‘brute’ - предполагает, что мы будем искать решение методом полного перебора
# n_neighbors = 20 - по скольким соседям ведется обучение
# n_jobs = -1 - вычисления будут вестись на всех свободных ядрах процессора
knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
# обучим модель
knn.fit(csr_data)


### Итак, мы готовы рекомендовать блюдо для еды ###
#  введем изначальные параметры: количество рекомендаций
#  и на основе какого блюда мы их хотим получить
recommendations = 10
search_food = 'Чизбургер'

# теперь найдем индекс блюда в матрице предпочтений.
# для начала найдем блюдо в заголовках датафрейма foods
food_search = foods[foods['title'].str.contains(search_food)]
#print(food_search)

# для простоты всегда будем брать первый вариант
# через iloc[0] мы берем первую строку столбца ['foodId']
food_id = food_search.iloc[0]['foodId']
#print('before =', food_id)

# далее по индексу блюда в датасете foods найдем соответствующий
# индекс в матрице предпочтений
food_id = user_matrix[user_matrix['foodId'] == food_id].index[0]
#print('after =', food_id)

# Это индекс блюда в матрице предпочтений(после того как мы сбросили индекс).
# Далее с помощью метода .kneighbors() найдем индексы ближайших соседей «Матрицы»
# csr_data[food_id], то есть индекс нужного нам блюда из матрицы предпочтений в формате сжатого хранения строкой
# n_neighbors - количество соседей(или рекомендаций).
# Обратите внимание, мы добавляем «лишнего» соседа (+1) из-за того, что алгоритм также считает расстояние до самого себя
distances, indices = knn.kneighbors(csr_data[food_id], n_neighbors=recommendations+1)
#print('before distances =', distances)
#print('before indices =', indices)

# На выходе мы получаем массив индексов блюд(indices) и массив расстояний(distances) до них.
# Для удобства преобразуем эти массивы в списки, а затем попарно объединим и создадим кортежи(tuples)
# уберем лишние измерения через squeeze() и преобразуем массивы в списки с помощью tolist()
indices_list = indices.squeeze().tolist()
distances_list = distances.squeeze().tolist()
#print('after distances_list =', distances_list)
#print('after indices_list =', indices_list)

# далее с помощью функций zip и list преобразуем наши списки в набор кортежей (tuple)
indices_distances = list(zip(indices_list, distances_list))
#print('indices_distances =', type(indices_distances[0]))

# и посмотрим на первые три пары/кортежа
#print('indices_distances(3) =', indices_distances[:3])

# остается отсортировать список по расстояниям через key = lambda x: x[1](то есть по второму элементу)
# в возрастающем порядке reverse = False
ind_dist_sorted = sorted(indices_distances, key=lambda x: x[1], reverse=False)
#print('sorted =', ind_dist_sorted)

# и убрать первый элемент с индексом 33(потому что это и есть "Чизбургер")
ind_dist_sorted = ind_dist_sorted[1:]
#print('sorted without 1 index =', ind_dist_sorted)

# Итак, индексы у нас есть. Теперь нужно найти какие блюда(названия) им соответствуют.
# Для этого обратимся к датафрейму foods
# создаем пустой список, в который будем помещать название блюда и расстояние до него
recom_list = []

# теперь в цикле будем поочередно проходить по кортежам
for ind_dist in ind_dist_sorted:

    # искать foodId в матрице предпочтений
    matrix_food_id = user_matrix.iloc[ind_dist[0]]['foodId']

    # выяснять индекс этого блюда в датафрейме foods
    id = foods[foods['foodId'] == matrix_food_id].index

    # брать название блюда и расстояние до него
    title = foods.iloc[id]['title'].values[0]
    dist = ind_dist[1]

    # помещать каждую пару в словарь
    # который, в свою очередь, станет элементом списка recom_list
    recom_list.append({'Title': title, 'Distance': dist})

#print('recom_list =', recom_list[0])

# Остается преобразовать наш список в датафрейм
# индекс будем начинать с 1, как и положено рейтингу
recom_df = pd.DataFrame(recom_list, index=range(1, recommendations+1))
print(recom_df)

### Сегодня мы создали коллаборативную рекомендательную систему, которая предлагает блюда на основе предпочтений пользователей ###


































