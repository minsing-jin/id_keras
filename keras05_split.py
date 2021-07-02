from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. Data
x = np.array(range(1,101))
y = np.array(range(101,201))

x_train = x[:60]
x_val = x[60:80]
x_test = x[80:]

y_train = y[:60]
y_val = y[60:80]
y_test = y[80:]

model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=130, batch_size=1, validation_data=(x_val, y_val))

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

result = model.predict([101,102,103])
print('result : ', result)