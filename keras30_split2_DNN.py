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

x_pred = [[6,7,8,9,10]]


# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
# model.add(LSTM(10, input_shape = (3,1), return_sequences=True))    #LSTM
# model.add(Dense(100))
model.add(Dense(64, input_shape = (5,), activation = 'relu'))
model.add(Dense(32,activation = 'relu'))
model.add(Dense(1000, activation = 'relu'))
model.add(Dense(1))


# 3. 컴파일, 훈련
# 4. 평가, 예측

# 실습, Dense로 만드시오!
# 