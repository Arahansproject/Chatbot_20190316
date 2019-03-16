# K-최근접 이웃 (k-Nearest Neighbors)

import mglearn
import matplotlib.pyplot as plt

mglearn.plots.plot_knn_regression(n_neighbors=1)

# n_neighbors=1 일 때 알고리즘

#plt.show()

"""
x축에 3개의 테스트 데이터를 녹색별로 표시합니다.
최근접 이웃을 한개만 사용할 때 예측은 가장 가까운 데이터의
포인트의 값으로 인식합니다. 
"""

# n_neighbors=3 일 때 알고리즘

mglearn.plots.plot_knn_regression(n_neighbors=3)

#plt.show()

from mglearn.datasets import make_wave
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

x, y = make_wave(n_samples=40)

x_train, x_test, y_train, y_test = \
    train_test_split(x, y, random_state=0, test_size=0.3)  # 랜덤값을 0에 고정하기 위해 random_state = 0

knn_reg = KNeighborsRegressor(n_neighbors=3, n_jobs=-1)
# n_jobs = 사용할 코어의 수. -1이면 모든 코어를 사용함
knn_reg.fit(x_train, y_train)

print("{:3f}".format(knn_reg.score(x_test, y_test)))

# 0.697183 정확도가 70%가 안되므로 추가 작업이 필요