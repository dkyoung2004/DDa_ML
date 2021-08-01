
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time


#기본 데이터셋 설정 난수로 할 수있으나, 직관적으로 기울기의 적절성을 보기위하여 기울기가 1이고 y절편이 0인 데이터셋을 만들어옴
x = [1,2,3,4,5,6,7,8]
y = [1,2,3,4,5,6,7,8]
x = np.array(x)
y = np.array(y)
#초기 기울기와 y절편을 랜덤으로 설정
#rmse방식으로 할 수도 있었으나 우선은 이렇게 설정.
W = np.random.rand()
b =np.random.rand()
def decent(x,y,W,b,lr):
    #일차함수에서 비롯된 y의 값과 본 데이터셋의 y값의 차이를 제곱하고, 데이터의 개수만큼 나눠주는데, 미분을 하면
    #지수가 계수로 가는 것을 인지하고 2로도 한번 더 나눠줌
    cost = np.sum((W*x+b-y)**2)/len(x)
    gradient_W =np.sum((W*x+b-y)*x)/len(x) #위에 코스트 함수를 미분한 것
    gradient_b =np.sum((W*x+b-y)*2)/len(x) #y값의 오차 만 봄 어차피 더하기 빼기만 하면되니께
    W -= lr*gradient_W
    b -= lr*gradient_b
    return cost,W
lr =0.001

for i in range(1,800):
    cost,W =decent(x,y,W,b,lr)
    print("W:",W,"cost:",cost)
plt.plot(x,y)
plt.plot(x,(W*x+b),color="red")
plt.show()
"""
def animate(x,y,W,b,lr):
    cost,W,b = decent(x,y,W,b,lr)
    x1 =x
    y2 =W*x1+b
    line.set_data(x1,y2)
    return line,

def init():
    line.set_data([],[])
    return line,
ani = FuncAnimation(figure,animate(x,y,W,b,lr),init_func=init,frames=100,interval=10,blit=True)
ax.plot(x,y,color = "red")
ani.save('exAnimation.gif', writer='imagemagick', fps=30)
plt.show()

for i in range (1,300):
    cost,W = decent(x,y,W,b,lr)
    print("cost:",cost,"W",W)
for p in range(200):
    W1,b1=decent(x,y,W,b,lr)
    line.set_xdata(x)
    line.set_ydata(W1*x+b1)
    figure.canvas.draw()

    figure.canvas.flush_events()
    time.sleep(0.1)
"""

