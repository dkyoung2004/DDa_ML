import numpy as np
import tensorflow as tf

x= np.array([[1, 2, 1, 1], [2, 1, 3, 2], [3, 1, 3, 4], [4, 1, 5, 5], [1, 7, 5, 5], 
                                                        [1, 2, 5, 6], [1, 6, 6, 6], [1, 7, 7, 7]])
y = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]]) 
learning_rate = 0.01
W = tf.Variable(tf.random.normal([8, 4]))
b = tf.Variable(tf.random.normal([8,3]))
def softmax(W,x,y):
    
    return 0
def predict(x):
    #행렬곱하는 함수(벡 스칼라의 곱)
    z = tf.matmul(x, W) + b
    hypothesis = 1 / (1 + tf.exp(-z))
    return hypothesis
hypothesis = predict(x) 
cost= tf.reduce_mean(-tf.reduce_sum(y*tf.math.log(hypothesis) + (1-y)*tf.math.log(1-hypothesis)))
W_grad, b_grad = tape.gradient(cost, [W, b])
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000+1):
        sess.run(optimizer,feed_dict={X:x,Y: y})
        if

