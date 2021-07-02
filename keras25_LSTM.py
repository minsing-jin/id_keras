# Dense layer는 가장 기초적인 노드구성
# 그다음 이미지 조각조각 CNN을 했음-> mnist
# inputshape와 outputshape 히든레이어 잘 조정
# 우리가 돈벌이 할 수 있는것
# 1. 이미지
# 2. 시계열 -> time series =시간의 흐름 => 비트코인, 미세먼지와 같은 시간순서대로 데이터 나열

import numpy as np


x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]])
y = np.array([4,5,6,7])
print(x.shape)  #(4, 3)
print(y.shape)  #(4,)

x = x.reshape(4,3,1)    #데이터 내용은 바뀌지 않고 형태만 바뀜
print(x.shape)    #인풋 맞는데 어디서 알아본거이?


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(10, input_shape = (3,1)))    #LSTM

model.summary()


# 과제
# RNN에 파라미타가 왜 저렇게 나오는지 확인


# shape LSTM에서 맞추는법 잘 모르겠음.