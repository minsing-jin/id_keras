import numpy as np

# 1. 데이터
from sklearn.datasets import load_iris
dataset = load_iris()
x = dataset.data
y = dataset.target
print(x[:5])
print(y[:5])
print(x.shape, y.shape) # (150, 4) (150,)

## 원핫인코딩 OneHotEncoding
# from tensorflow.keras.utils import to_categorical

# print(y[145:])
# y = to_categorical(y)          # (150,) --> (150,3)
# print(y[145:])


# print(dataset.feature_names)
# print(dataset.DESCR) # 회귀 문제

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=66
)
print(x_test[0])

# 2. 모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

input1 = Input(shape=(4,))
h1 = Dense(10)(input1)
h2 = Dense(10)(h1)
h3 = Dense(10)(h2)
h4 = Dense(5)(h3)
output1 =  Dense(3, activation='softmax')(h4)
model = Model(inputs=input1, outputs=output1)

# 3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, batch_size=1, epochs=100)

# 4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])
# print('results : ', results)

# y_predict = model.predict(x_test)
# print("input: ",x_test[:5])
# print("GT: ", y_test[:5])
# print("predict: ", y_predict[:5])

# 스파스 카테고리칼 크로스엔트로피까지 하면 원핫인코딩이 필요없다
# 어떤걸 선택하던간에 시스템은 원핫을 시켜놓고함.