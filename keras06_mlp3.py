import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. 데이터
x = np.array([[10, 85, 70],[90, 85, 100],[80, 50, 30], [43, 60, 100]]) # (4, 3)
y = np.array([75, 65, 33, 65]) # (4, )

# 2. 모델구성
model = Sequential()
model.add(Dense(10, input_shape=(3,)))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(1))

# 3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam', metrics=['mse'])
# model.compile(loss='mse', optimizer='adam', metrics=['mae'])
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy']) # 회귀, 분류 중 분류에 쓰임. 예상 출력값과 100% 맞았을 때 true. 회귀에서는 0이 나올 수 밖에 없음.
model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae']) # 여러 평가지표를 넣을 수 있음.
model.fit(x, y, batch_size=1, epochs=10)

# 4. 평가, 예측
loss = model.evaluate(x, y)
print("loss : ", loss)

# Epoch 10/10
# 4/4 [==============================] - 0s 6ms/step - loss: 36.2823 - mse: 36.2823 - mae: 4.6923
# 1/1 [==============================] - 0s 119ms/step - loss: 36.6343 - mse: 36.6343 - mae: 4.1494
# loss :  [36.634307861328125, 36.634307861328125, 4.149420261383057]

# Epoch 10/10
# 4/4 [==============================] - 0s 3ms/step - loss: 37.6766 - accuracy: 0.0000e+00
# 1/1 [==============================] - 0s 109ms/step - loss: 38.2353 - accuracy: 0.0000e+00
# loss :  [38.23528289794922, 0.0]