import numpy as np
import tensorflow as tf
 
# 학습 데이터
x_data = [
    [1, 2],
    [2, 3],
    [3, 1],
    [4, 3],
    [5, 3],
    [6, 2]
]
 
y_data = [
    [0],
    [0],
    [0],
    [1],
    [1],
    [1],
]
 
# placeholder는 데이터셋을 세팅 한다.
#X은 실수형 자료형으로 [모르는구조, 2]의 행렬을 세팅하고
#Y은 실수형 자료형으로 [모르는구조, 1]의 행렬을 세팅한다.
X = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
Y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
#w는 가중치로 랜덤한 값을 부여한다.
#b고정된 편향치를 랜덤하게 부여한다.
W = tf.Variable(tf.random.normal([2, 1]), 'weight')
b = tf.Variable(tf.random.normal([1]), 'bias')
 
# 가설
hypothesis = tf.sigmoid(tf.linalg.matmul(X, W) + b)
 
# cost function
cost = -tf.reduce_mean(Y * tf.math.log(hypothesis) + (1 - Y) * tf.math.log(1 - hypothesis))
 
# cost function
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)
 
# True if hypothesis > 0.5 else false
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
# 정확도 체크
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
 
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
 
    for step in range(10001):
        cost_val, train_val = sess.run(
            [cost, train],
            feed_dict={X: x_data, Y: y_data}
        )
 
        if step % 200 == 0:
            print(step, "\tcost : ", cost_val)
 
    # Accuracy report
    hypo_val, predict_val, acc_val = sess.run(
        [hypothesis, predicted, accuracy],
        feed_dict={X: x_data, Y: y_data}
    )
    print("\nHypothesis : ", hypo_val, "\nCorrect(Y) : ", predict_val, "\nAccuracy : ", acc_val)
