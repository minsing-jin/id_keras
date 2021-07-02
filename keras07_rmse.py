import numpy as np
from tensorflow.python.keras import activations

# 1. data
x = np.array([[1,2,3,4,5,6,7,8,9,10],
            [11,12,13,14,15,16,17,18,19,20]]) # shape : (2, 10)
y = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.transpose(x) # shape : (10, 2), matching dimension
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.1
)
# 2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
#model.add(Dense(10, input_dim=2))
model.add(Dense(10, input_shape=(2,))) # ignore row (ex) (1000, 10, 10, 3) -> input_shape=(10,10,3)
model.add(Dense(5))
model.add(Dense(1))

# 3. compile, trainnig
model.compile(loss='mse',optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, batch_size=1, epochs=100, 
    validation_split=0.2)

# 4. evaluate, predict
results = model.evaluate(x_test, y_test)
print('results : ', results)

# predict [[11,12,13],[21,22,23]]
y_predict = model.predict(np.transpose([[11,12,13],[21,22,23]]))
print('y_predict : ', y_predict)

from sklearn.metrics import mean_squared_error
def rmse(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print('rmse : ', rmse(y_test, model.predict(x_test)))

# Epoch 100/100
# 7/7 [==============================] - 0s 4ms/step - loss: 2.3141e-04 - mse: 2.3141e-04 - val_loss: 2.3748e-04 - val_mse: 2.3748e-04
# 1/1 [==============================] - 0s 68ms/step - loss: 3.5096e-04 - mse: 3.5096e-04
# results :  [0.00035096192732453346, 0.00035096192732453346]
# y_predict :  [[10.972466]
#  [11.966686]
#  [12.960899]]
# rmse :  0.018733978271484375