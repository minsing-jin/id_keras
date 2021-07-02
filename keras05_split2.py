import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. data
x = np.array(range(1,101))
y = np.array(range(101,201))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.4, train_size=0.6
)
x_test, x_val, y_test, y_val = train_test_split(
    x_test, y_test, test_size=0.5
)

print(x_train)
print(x_val)
print(x_test)
print(x_train.shape)
print(x_val.shape)
print(x_test.shape)

# 2. model
model = Sequential();
model.add(Dense(10, activation='relu', input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))
 
# 3. compile, training
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, batch_size=1, epochs=60, 
    validation_data=(x_val, y_val))

# 4. evaluate, predict
y_predict = model.predict([101, 102, 103])
print('y_predict : ', y_predict)