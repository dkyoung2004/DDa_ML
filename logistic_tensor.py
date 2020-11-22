import numpy as np
import tensorflow as tf
 
# 학습 데이터
x_data = np.array([
    [1., 2.],
    [2., 3.],
    [3., 1.],
    [4., 3.],
    [5., 3.],
    [6., 2.]],dtype= np.float32)
 
y_data = np.array([
    [0.],
    [0.],
    [0.],
    [1.],
    [1.],
    [1.],
],dtype=np.float32)

x_test = np.array([[3., 0.],
                   [4., 1.]],
                   dtype=np.float32)

y_test = np.array([[0.],
                   [1.]],
                   dtype=np.float32)
# placeholder는 데이터셋을 세팅 한다.
#w는 가중치로 랜덤한 값을 부여한다. 여기서 가중치란? 입력된 변수가 결과출력에 주는 영향도를 조절하는 매개변수.
#b고정된 편향치를 랜덤하게 부여한다. 편향치는 데이터가 얼마나 잘 활성화가 되는냐를 저정하는 매개변수이다.
W = tf.Variable(tf.random.normal([2, 1]))
b = tf.Variable(tf.random.normal([1]))
 
# 가설
learning_rate = 0.01

# Hypothesis and Prediction Function
def predict(X):
    #행렬곱하는 함수(벡 스칼라의 곱)
    z = tf.matmul(X, W) + b
    hypothesis = 1 / (1 + tf.exp(-z))
    return hypothesis

# Training
for i in range(2000+1):

    with tf.GradientTape() as tape:

        hypothesis = predict(x_data)
        cost = tf.reduce_mean(-tf.reduce_sum(y_data*tf.math.log(hypothesis) + (1-y_data)*tf.math.log(1-hypothesis)))        
        W_grad, b_grad = tape.gradient(cost, [W, b])

        W.assign_sub(learning_rate * W_grad)
        b.assign_sub(learning_rate * b_grad)

    if i % 400 == 0:
        print(">>> #%s \n Weights:%s \n Bias: %s \n cost: %s\n" % (i, W.numpy(), b.numpy(), cost.numpy()))

hypothesis = predict(x_test)
print("Prob: \n", hypothesis.numpy())
print("Result: \n", tf.cast(hypothesis > 0.5, dtype=tf.float32).numpy())
