import numpy as np
import tensorflow as tf

x= np.array([[1, 2, 1, 1], [2, 1, 3, 2], [3, 1, 3, 4], [4, 1, 5, 5], [1, 7, 5, 5], 
                                                        [1, 2, 5, 6], [1, 6, 6, 6], [1, 7, 7, 7]])
y = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]]) 
learning_rate = 0.01
W = tf.Variable(tf.random.normal([2, 1]))
b = tf.Variable(tf.random.normal([1]))
 
def softmax(W,x,y):

    return 0
def predict(x):
    #행렬곱하는 함수(벡 스칼라의 곱)
    z = tf.matmul(x, W) + b
    hypothesis = 1 / (1 + tf.exp(-z))
    return hypothesis
