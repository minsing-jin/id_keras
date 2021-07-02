# 16-2 카피

import numpy as np
from tensorflow.keras.datasets import mnist


# 1.데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape, x_train.shape)   #(60000, 28, 28) (60000, 28, 28)
# print(y_train.shape, y_test.shape)    #(60000,) (10000,)

print(x_train[0])
print(y_train[0])





# 2. 모델
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = load_model("./keras/Model/k23_1_model_1.h5")
# model = load_model(""./keras/Model/k23_1_model_2.h5")
model.summary()
 


'''
model = Sequential()
model.add(Conv2D(filters=30, kernel_size=(2,2), padding = 'same',
                strides =1, input_shape = (28,28,1)))
model.add(Conv2D(20, (2,2), activation='relu'))
model.add(Conv2D(20, (2,2)))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.save('./keras/model/k23_1_model1_1.h5')




# 3. 컴파일, 훈련
model.compile(loss='sparse_catagorical_crossentropy', optimizer='adam', metrics=['acc'])
# modle.compile(loss='catagoricl_corssentropy', optimizer='adam', matrics=['acc'])
model.fit(x_train, y_train, epochs=10, validation_split=0.2, verbose=1)

model.save('./keras/Model/k23_1_model_2.h5')

'''
# 4. 평가, 예측
results = model.evaluate(x_test, y_test)

print("loss : ", results[0])
print("acc : ", results[1])


# 3번부터 전부 주석처리














