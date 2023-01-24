import pandas as pd

foods = pd.read_csv('recom/foods.csv')
countbuys = pd.read_csv('recom/countbuys.csv')
#print(foods.head(5))
#print(countbuys.head(5))

user_matrix = countbuys.pivot(index='foodId', columns='userId', values='countbuy')
#print(user_matrix.head())
user_matrix.fillna(0, inplace=True)
#print(user_matrix.head())

buy_votes = countbuys.groupby('foodId')['countbuy'].agg('max')
print(buy_votes)

buy_mask = buy_votes[buy_votes > 6].index
print(buy_mask)

user_matrix = user_matrix.loc[buy_mask]
#print(user_matrix.shape)
#print(user_matrix.head(8))

buy_list = []

for buy_n in buy_mask:
    buy_list.append(buy_n)

print('buy_list =', buy_list)

recom_list = []

for ind_dist in buy_list:
    title = foods.iloc[ind_dist-1]['title']
    recom_list.append({'Title': title})

print("\n".join(map(str, recom_list)))

