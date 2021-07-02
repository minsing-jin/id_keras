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
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, batch_size=1, epochs=10)

# 4. 평가, 예측
loss = model.evaluate(x, y)
print("loss : ", loss)

# Epoch 10/10
# 4/4 [==============================] - 0s 6ms/step - loss: 40.2323
# 1/1 [==============================] - 0s 132ms/step - loss: 28.5052
# loss :  28.505237579345703