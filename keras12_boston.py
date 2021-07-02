import numpy as np

# 1. 데이터
from sklearn.datasets import load_boston
dataset = load_boston()
x = dataset.data
y = dataset.target
print(x[:5])
print(y[:5])
print(x.shape, y.shape) # (506, 13) (506,)

print(dataset.feature_names)
print(dataset.DESCR) # 회귀 문제

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=66
)
print(x_test[0])

# 2. 모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

input1 = Input(shape=(13,))
h1 = Dense(10)(input1)
h2 = Dense(10)(h1)
h3 = Dense(10)(h2)
h4 = Dense(5)(h3)
output1 =  Dense(1)(h4)
model = Model(inputs=input1, outputs=output1)

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, batch_size=5, epochs=100)

# 4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('results : ', results)
