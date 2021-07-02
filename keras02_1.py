import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. Data
x_train = np.array([1,2,3,4,5])
y_train = np.array([1,2,3,4,5])

x_test = np.array([6,7,8])
y_test = np.array([6,7,8])

# 2. model
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(3))
model.add(Dense(10))
model.add(Dense(4))
model.add(Dense(1))


# 3. compile, training
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, batch_size=1, epochs=100)

# 4. evaluation
loss = model.evaluate(x_test,y_test)
print('loss: ', loss)

results = model.predict([9])
print('results: ', results)
