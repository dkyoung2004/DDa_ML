import numpy as np
import tensorflow as tf

x=  np.array([[1, 2, 1, 1], [2, 1, 3, 2], [3, 1, 3, 4], [4, 1, 5, 5], [1, 7, 5, 5], 
                                                        [1, 2, 5, 6], [1, 6, 6, 6], [1, 7, 7, 7]])
#값설정
y = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]]) 
# 라벨링
learning_rate = 0.01
W = tf.Variable(tf.random.normal([8, 4]))
b = tf.Variable(tf.random.normal([8,3]))
#초기는 랜덤한 가중치 행렬 부여
##변수 생성해서 메모리에 저장

def softmax(W,x,y):
    
    return 0
def predict(x):
    #행렬곱하는 함수(벡 스칼라의 곱)
    z = tf.matmul(x, W) + b
    hypothesis = 1 / (1 + tf.exp(-z))
    return hypothesis
hypothesis = predict(x) 
cost= tf.reduce_mean(-tf.reduce_sum(y*tf.math.log(hypothesis) + (1-y)*tf.math.log(1-hypothesis)))
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
train_op = optimizer.minimize(cost)
for i in range(200+1):
    if i %10 == 0:
        print('epoch:{}/{}, loss:{:.4f}'.format(i,200,cost))

