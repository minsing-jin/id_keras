import numpy as np

# 1. 데이터
x = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10],
                [9, 10, 11], [10, 11, 12], [20,30,40], [30,40,50], [40,50,60]])

y = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70])
y_pred = np.array([50,60,70])
print(x.shape)      #(13, 3)
print(y.shape)       #(13,)

x = x.reshape(13,3,1)    #(13, 3, 1)
print(x.shape)

# 실습, 코딩하시오, lstm
# 내가원하는건 80!
# reshape할 것

# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(10, input_shape = (3,1)))    #LSTM
model.add(Dense(100))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1))


# model.summary()


# 과제
# RNN에 파라미타가 왜 저렇게 나오는지 확인


# shape LSTM에서 맞추는법 잘 모르겠음.

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=1000, batch_size=1)


# 4. 평가, 예측
results = model.evaluate(x,y)
print('loss : ', results)

x_pred = np.array([50,60,70])  #(3, ) -> (1.3.1) [[[50], [60], [70]]]
x_pred = x_pred.reshape(1,3,1)

y_pred = model.predict(x_pred)
print(y_pred)

# 회귀모델이며 결과는 숫자로 나올것임

