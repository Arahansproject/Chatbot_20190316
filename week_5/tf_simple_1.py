import tensorflow as tf
import matplotlib.pyplot as plt

tf.set_random_seed(777)

X = [1, 2, 3] # 확률 변수는 주로 대문자로 설정, 1,2,3 중에 하나가 들어갈 수 있으므로, x = 3
Y = [1, 2, 3]

W = tf.placeholder(tf.float32)
hypothesis = X * W

cost = tf.reduce_mean(tf.square(hypothesis - Y))
sess = tf.Session()

W_history = []
cost_history = []

for i in range(-30, 50):
    curr_W = i * 0.1
    curr_cost = sess.run(cost, {W: curr_W})
    W_history.append(curr_W)
    cost_history.append(curr_cost)
# 차트로 확인
plt.plot(W_history, cost_history)
plt.show()