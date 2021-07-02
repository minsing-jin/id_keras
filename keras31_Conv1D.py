import numpy as np
from tensorflow.keras.datasets import mnist


# 1.데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, x_test.shape)   #(60000, 28, 28) (10000, 28, 28)
print(y_train.shape, y_test.shape)    #(60000,) (10000,)

# x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.      #xtrain 데이터는 0에서1사이 값으로 바꾸기 위해 실수형으로 바꿔주고/255로 나누어줌  # RGB까지 reshape해줌.
# x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.
# print(x_train[0])

x_train = x_train.reshape.astype('float32')/255.      #xtrain 데이터는 0에서1사이 값으로 바꾸기 위해 실수형으로 바꿔주고/255로 나누어줌  # RGB까지 reshape해줌.
x_test = x_test.reshape.astype('float32')/255.



# x_train = x_train.reshape(60000,14,14,4)
# x_train = x_train.reshape(60000,784)


# 2. 모델
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(filters=100, kernel_size=(2,2), padding = 'same',#첫번째 layer는 input layer과 shape를 명시해줘야한다.(N,28,28,30) -> 3차원이지만 4차원으로 데이터를 명시
                strides =1, input_shape = (28,28,1))) 
                           # color는 필터와 같은것임, shape에서 맨 마지막 스칼라부분
# model.add(Conv2D(filters=30, kernel_size=2, padding = 'same',
#                     strides=1, input_shape=(28,28)))

model.add(Conv2D(20, (2,2), activation='relu'))
model.add(Conv2D(20, (2,2)))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='softmax'))
 


# 이미지 가지고도 DNN모델이 가능하다. 즉 가로세로 shape만 잘 맞으면 된다!
# numpy는 소수점 연산에 빠르다
# 데이터 전처리 -> 정규화(min, max, sclar)

# 3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics = ['acc'])     #원핫인코딩 안하면 스파스 카테고리칼 크로스 엔트로피
model.fit(x_train, y_train, epochs=20, validation_split=0.2, verbose=1)

# 4. 평가, 예측
results = model.evaluate(x_test, y_test)

print("loss : ", results[0])
print("acc : ", results[1])

# batch size를 쓰면 하나씩 훈련할꺼 10개씩 한꺼번에 훈려낙능