import numpy as np


X=[1,2,3,4,5,6]
Y=[0,0,0,1,1,1]

X = np.array(X)
Y = np.array(Y)
lr = 0.01
W = np.random.rand()
b =np.random.rand()
#log()은 밑이 e인 로그이다.
#log e X를 미분하면 1/X가 나온다.
#따라서 우리는 위 식을 미분하면 1/W*X+b 이 나온다. 
def logistic_recent(X,Y,W,b,lr):
    #cost = np.sum(-1*Y*np.log(W*X+b)-(1-Y)*np.log(1-(W*X+b)))/len(X)
    gradient_W = -1*(np.sum(Y*(1/W*X+b)+(1-Y(1/W*X+b)))/len(X))
    gradient_b = -1*(np.sum(Y*(1/W*X+b)+(1-Y(1/W*X+b)))/len(X))
    W -= lr*gradient_W
    b -= lr*gradient_b
    return W,b

for i in range(1,400):
    W,b = logistic_recent(X,Y,W,b,lr)
    print("W:",W,"b:",b)
