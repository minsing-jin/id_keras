import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

# 1
x = np.array(range(100))
y = np.array([range(711,811), range(100)])
print(x.shape) # (5, 100)
print(y.shape) # (2, 100)

x = np.transpose(x)
y = np.transpose(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=66
)

# 2
input1 = Input(shape=(5,))
xx = Dense(3)(input1)
xx = Dense(4)(xx)
output1 = Dense(2)(xx)
model = Model(inputs=input1, outputs=output1)
model.summary()

# 3
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, batch_size=1, epochs=100, verbose=5)