import pandas as pd
import os
import matplotlib.pyplot as plt
import mglearn
import numpy as np

ram_price = pd.read_csv(os.path.join(mglearn.datasets.DATA_PATH,"ram_price.csv"))

plt.semilogy(ram_price.date, ram_price.price)
plt.xlabel("년")
plt.ylabel("가격")
plt.show()

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

data_train = ram_price[ram_price['date'] < 2000]
data_test = ram_price[ram_price['date'] >= 2000]

x_train = data_train['date'][:,np.newaxis] # train data를 1열로 만듭니다
y_train = np.log(data_train['price'])
# 객체 생성시에 대문자로 오면 ()넣기 - 생성자 처리
tree = DecisionTreeRegressor().fit(x_train, y_train)
lr = LinearRegression().fit(x_train, y_train)

# test는 모든 데이터에 대해 적용합니다.
x_all = ram_price['date'].values.reshape(-1, 1) # x_all 을 1열로 만듭니다.

pred_tree = tree.predict(x_all)
price_tree = np.exp(pred_tree) # Log 값 되돌리기

pred_lr = lr.predict(x_all)
price_lr = np.exp(pred_lr) # log 값 되돌리기

plt.semilogy(ram_price['date'], price_tree, \
             label="tree predict", ls='-', dashes=(2,1))
plt.semilogy(ram_price['date'], price_lr, \
             label="linear regression", ls=':')
plt.semilogy(data_train['date'], data_train['price'], label='train data', alpha=0.4)
plt.semilogy(data_test['date'], data_test['price'], label='test data')

plt.legend(loc=1)
plt.xlabel('year', size=15)
plt.ylabel('price', size=15)
plt.show()

"""
램가격 데이터로 만든 리니어 모델과 트리모델의 예측값 비교
리니어모델은 직선으로 데이터를 근사합니다.
트리모델은 트레인세트을 완벽하게 피팅했습니다. 

그러나 트리모델은 트레인세트를 넘어가 버리면 마지막 포인트를 이용해
예측하는 것이 전부입니다. (tree predict가 직선이 되는 지점)

결정 트리의 주요 단점은 선가지치기(pre-pruning)를 함에도 불구하고
오버피팅되는 경향이 있어 일반화 성능이 좋지 않습니다. 
이에 대안으로 앙상블 방법을 많이 사용합니다.
"""





