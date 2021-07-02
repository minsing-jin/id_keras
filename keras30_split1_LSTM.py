import numpy as np

a = np.array(range(1,11))
size = 5
print(a)

def split_x(seq, size):
    aaa = []        
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i+size)]
        aaa.append(subset) 
    print(type(aaa))
    return np.array(aaa)

dataset = split_x(a,size)
print(dataset)


x = dataset[:,:size-1]
y = dataset[:, size-1]

print(x)
print(y)




# 2. 모델
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, LSTM

model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(1,4)))
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=1000, batch_size=1)


# 4. 평가, 예측
results = model.evaluate(x,y)
print('loss : ', results)
x_pred = np.array([[6,7,8,9,10]]) 
x_pred = x_pred.reshape(5,1)
y_pred = model.predict(x_pred)
print(y_pred)


# 실습, LSTM으로 만드시오!
# 