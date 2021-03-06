import tensorflow as tf
import numpy as np
import keras 
import pandas as pd
from keras.layers import Dense, Dropout, Flatten,BatchNormalization,Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D,AveragePooling2D
from keras.models import Sequential
from PIL import Image
from os import listdir
import numpy as np
image_directory = "./photos/"

def read_image_grayscale(file_name):
  return np.array(list(map(lambda pix: pix / 255, np.array(Image.open(image_directory + file_name).convert('L'), 'uint8'))))
input_y = pd.read_csv("./TCEAI_Contest/y_input.csv",sep=",")
print(input_y.head())
input_y = pd.Series(input_y['male'])
print(input_y)
x_train =  12
# x_test = 
y_train = np.array(input_y)
# y_test =
#Conv2D         합성곱을 할 차원의 수 flatten을 사용해 1차원이 되었으면 Conv1D가 될 것.

#stride         이동하는 간격
#kernal size    필터사이즈
#padding        유효한 값이 있는부분만 처리하는 지의 여부
#activation     활성화 함수 -ps-  현재는 이진분류를 하기때문에 최종 출력층에서는 sigmoid 사용예정
#filters        필터의 층 개수

model = Sequential()
model.add(Conv2D(filters=12,kernel_size=(20, 20),strides=(2, 2),padding='valid',input_shape=(480, 480, 1)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2)))

model.add(Conv2D(filters=24,kernel_size=(5, 5),strides=(1,1),activation='relu',padding='valid'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2)))

model.add(Conv2D(filters=24,kernel_size=(3, 3),strides=(1,1),activation='relu',padding='valid'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2)))
#신경망 시작
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64,activation='relu'))
model.add(Dense(2,activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
#history = model.fit(x_train,y_train,batch_size=512,epochs=30,verbose=1)