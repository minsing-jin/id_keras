# 라인 삭제 단축키 -> 쉬프트 딜리트
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D

#이진분류-> 다중분류로 수정하시오
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.utils import validation

#1. 데이터
data = load_breast_cancer()
x = data.data
y = data.target

#print(x.shape, y.shape)
#print(data.feature_names)
#print(data.DESCR)
#print(x[:5])
#print(y[:5])

# 원 핫 인코딩
# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)          # (150,) --> (150,3)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=66
)


#2 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

model = Sequential()
model.add(Dense(10, activation = 'relu', input_shape=(30,)))
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(2, activation = 'softmax')) # 레이어 통과값 0~1 사이로 수렴됨.

#3
# model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['acc'])
# model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['acc'])

# early stopping

model.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min')   #auto로 할 경우 알아서 loss일때면 min이 잡힘
                                        # loss                        auto, min, max            
hist = model.fit(x_train, y_train, epochs=1000, validation_split=0.2,   #validation split는 traindata에서 빼옴
            callbacks = [early_stopping]
)



#4. 
result = model.evaluate(x_test, y_test)
print("loss : ", result[0])
print("acc : ", result[1])

y_predict = model.predict(x_test[-5:-1])
print(y_predict) 
print(y_test[-5:-1])
# # print("Input : ", x_test[:5])
# print("TrueOutput : ", y_test[:5])
# print("PredOutput : ", y_predict[:5])

# 밑에 실행결과 잘 붙여놓을것!

# 수업끝나면 트레인테스트 스플릿할것


# loss로 얼리 스탑을 하는게 좋음. acc보다

#  질문
# 1. val loss를 왜 earrly stopping  하는지 잘 모르겠음
# 2. 결과값을 어떻게 분석하누

print(hist)
print(hist.history.keys())
# print(hist.history['loss'])
print(hist.history['val_loss'])


# 시각화/그래프
import matplotlib.pyplot as plt
# plt.plot(hist.history['loss'])
# plt.plot(hist.history['val_loss'])
# matrix에 있는 것들은 괜찮음
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])

plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train loss', 'val_loss'])
plt.show()