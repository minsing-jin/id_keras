import numpy as np
import tensorflow as tf

#1. data
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(30, input_dim = 1))
model.add(Dense(40))
model.add(Dense(20))
model.add(Dense(1))

#3. compile, training
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, batch_size=1, epochs=500)

#4. evaluation
loss = model.evaluate(x,y)
print('loss: ', loss)

results = model.predict([14])
print('results: ', results)

# mse
# batchsize default
# tunning
