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

from tensorflow.keras.utils import to_categorical

y = to_categorical(y)          # (150,) --> (150,3)


#2 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

model = Sequential()
model.add(Dense(10, activation = 'relu', input_shape=(30,)))
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(2, activation = 'softmax')) # 레이어 통과값 0~1 사이로 수렴됨.

#3
# model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['acc'])
# model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['acc'])

# early stopping
i = 0

model.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, batch_size=1, epochs=50, verbose=1, validation_split=0.1)

#4. 
result = model.evaluate(x_test, y_test)
print("loss : ", result[0])
print("acc : ", result[1]) 

y_predict = model.predict(x_test) 
# print("Input : ", x_test[:5])
print("TrueOutput : ", y_test[:5])
print("PredOutput : ", y_predict[:5])

# 밑에 실행결과 잘 붙여놓을것!

# 수업끝나면 트레인테스트 스플릿할것
