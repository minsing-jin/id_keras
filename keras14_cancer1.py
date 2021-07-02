#이진분류
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.utils import validation

#1. 데이터
data = load_breast_cancer()
x = data.data
y = data.target

#print(x.shape, y.shape)
#print(data.feature_names)
#print(data.DESCR)
#print(x[:5])
#print(y[:5])

from sklearn.model_selection import train_test_split
#x_train, y_train, x_test, y_test = train_test_split(
#    x, y, shuffle=True, train_size = 0.8, random_state = 66
#    )
# 충격. 순서대로 써야함!!!!!!!!!!!!!!!!!!!!! 
# x 먼저 다 쓰고. 그 다음 y 쓰기.
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=66
)

#2 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

model = Sequential()
model.add(Dense(10, activation = 'relu', input_shape=(30,)))
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(2, activation = 'sigmoid')) # 레이어 통과값 0~1 사이로 수렴됨.

#3
model.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, batch_size=1, epochs=100, verbose=4, validation_split=0.1)
#4. 
result = model.evaluate(x_test, y_test)
print("loss : ", result[0])
print("acc : ", result[1]) 

y_predict = model.predict(x_test) 
print("Input : ", x_test[:5])
print("TrueOutput : ", y_test[:5])
print("PredOutput : ", y_predict[:5])

# 밑에 실행결과 잘 붙여놓을것!

# 수업끝나면 트레인테스트 스플릿할것

# loss :  0.23053961992263794
# acc :  0.9035087823867798
# Input :  [[1.290e+01 1.592e+01 8.374e+01 5.122e+02 8.677e-02 9.509e-02 4.894e-02
#   3.088e-02 1.778e-01 6.235e-02 2.143e-01 7.712e-01
# 1.689e+00 1.664e+01
#   5.324e-03 1.563e-02 1.510e-02 7.584e-03 2.104e-02
# 1.887e-03 1.448e+01
#   2.182e+01 9.717e+01 6.438e+02 1.312e-01 2.548e-01
# 2.090e-01 1.012e-01
#   3.549e-01 8.118e-02]
#  [1.256e+01 1.907e+01 8.192e+01 4.858e+02 8.760e-02
# 1.038e-01 1.030e-01
#   4.391e-02 1.533e-01 6.184e-02 3.602e-01 1.478e+00
# 3.212e+00 2.749e+01
#   9.853e-03 4.235e-02 6.271e-02 1.966e-02 2.639e-02
# 4.205e-03 1.337e+01
#   2.243e+01 8.902e+01 5.474e+02 1.096e-01 2.002e-01
# 2.388e-01 9.265e-02
#   2.121e-01 7.188e-02]
#  [1.160e+01 1.284e+01 7.434e+01 4.126e+02 8.983e-02
# 7.525e-02 4.196e-02
#   3.350e-02 1.620e-01 6.582e-02 2.315e-01 5.391e-01
# 1.475e+00 1.575e+01
#   6.153e-03 1.330e-02 1.693e-02 6.884e-03 1.651e-02
# 2.551e-03 1.306e+01
#   1.716e+01 8.296e+01 5.125e+02 1.431e-01 1.851e-01
# 1.922e-01 8.449e-02
#   2.772e-01 8.756e-02]
#  [1.276e+01 1.337e+01 8.229e+01 5.041e+02 8.794e-02
# 7.948e-02 4.052e-02
#   2.548e-02 1.601e-01 6.140e-02 3.265e-01 6.594e-01
# 2.346e+00 2.518e+01
#   6.494e-03 2.768e-02 3.137e-02 1.069e-02 1.731e-02
#   4.391e-02 1.533e-01 6.184e-02 3.602e-01 1.478e+00 3.212e+00 2.749e+01         1 2.208e-01
#   9.853e-03 4.235e-02 6.271e-02 1.966e-02 2.639e-02 4.205e-03 1.337e+01
#   2.243e+01 8.902e+01 5.474e+02 1.096e-02 8.759e-021 2.002e-01 2.388e-01 9.265e-02
#   2.121e-01 7.188e-02]                  1 9.861e-01
#  [1.160e+01 1.284e+01 7.434e+01 4.126e+02 8.983e-02 7.525e-02 4.196e-02         3 1.568e-02
#   3.350e-02 1.620e-01 6.582e-02 2.315e-01 5.391e-01 1.475e+00 1.575e+01         1 2.196e-01
#   6.153e-03 1.330e-02 1.693e-02 6.884e-03 1.651e-02 2.551e-03 1.306e+01
#   1.716e+01 8.296e+01 5.125e+02 1.431e-01 1.851e-01 1.922e-01 8.449e-02
#   2.772e-01 8.756e-02]
#  [1.276e+01 1.337e+01 8.229e+01 5.041e+02 8.794e-02 7.948e-02 4.052e-02
#   2.548e-02 1.601e-01 6.140e-02 3.265e-01 6.594e-01 2.346e+00 2.518e+01
#   6.494e-03 2.768e-02 3.137e-02 1.069e-02 1.731e-02 4.392e-03 1.419e+01
#   1.640e+01 9.204e+01 6.188e+02 1.194e-01 2.208e-01 1.769e-01 8.411e-02
#   2.564e-01 8.253e-02]
#  [1.134e+01 2.126e+01 7.248e+01 3.965e+02 8.759e-02 6.575e-02 5.133e-02
#   1.899e-02 1.487e-01 6.529e-02 2.344e-01 9.861e-01 1.597e+00 1.641e+01
#   9.113e-03 1.557e-02 2.443e-02 6.435e-03 1.568e-02 2.477e-03 1.301e+01
#   2.915e+01 8.399e+01 5.181e+02 1.699e-01 2.196e-01 3.120e-01 8.278e-02
#   2.829e-01 8.832e-02]]
# TrueOutput :  [1 1 1 1 1]
# PredOutput :  [[0.20727727 0.7607147 ]
#  [0.1210365  0.84408885]
#  [0.08521867 0.8822005 ]
#  [0.16404802 0.801369  ]
#  [0.29339385 0.6840954 ]]