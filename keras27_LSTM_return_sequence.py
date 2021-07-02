import numpy as np

# 1. 데이터
x = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10],
                [9, 10, 11], [10, 11, 12], [20,30,40], [30,40,50], [40,50,60]])

y = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70])
y_pred = np.array([50,60,70])
print(x.shape)      #(13, 3)
print(y.shape)       #(13,)

x = x.reshape(13,3,1)    #(13, 3, 1)
print(x.shape)

# 실습, 코딩하시오, lstm
# 내가원하는건 80!
# reshape할 것

# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(10, input_shape = (3,1), return_sequences=True))    #LSTM
model.add(LSTM(10))    #return sequence는 순서를 다시 리턴한다. LSTM 두단쌓을때면 사용
model.add(Dense(100))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1))


# model.summary()


# 과제
# 함수 알아오기
# def split_x(seq, size):
    # aaa = []        
    # for i in range(len(seq) - size + 1):
    #     subset = seq[i : (i+size)]
    #     aaa.append(subset) 
    # print(type(aaa))
    # return np.array(aaa)

# 첫번째는 데이터 리스트 주는것


# def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    # series = tf.expand_dims(series, axis=-1)
    # ds = tf.data.Dataset.from_tensor_slices(series)
    # ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    # ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    # ds = ds.shuffle(shuffle_buffer)
    # ds = ds.map(lambda w: (w[:-1], w[1:]))
    # return ds.batch(batch_size).prefetch(1)

# 첫번째는 완전히 이해할것, 두번째꺼는 인풋과 아웃풋이 무엇인지 대강 어떤 놈인지정도-> 이거 잘 이해하면 다음수업 잘 이해할수 있음.
# 이게 어떤건지정도 알것, 그리고 파라미타 개수,

# shape LSTM에서 맞추는법 잘 모르겠음.

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=1000, batch_size=1)


# 4. 평가, 예측
results = model.evaluate(x,y)
print('loss : ', results)

x_pred = np.array([50,60,70])  #(3, ) -> (1.3.1) [[[50], [60], [70]]]
x_pred = x_pred.reshape(1,3,1)

y_pred = model.predict(x_pred)
print(y_pred)

# 회귀모델이며 결과는 숫자로 나올것임



# 과기부에서 진행하는 인공지능 온라인 경진대회
# 인공지능 빨리 할수록 좋음. 


# 요약 두단 연결 가능 컨볼루션 두단쌓으면 좋지만 lstm은 별로임. 잘 쓰면 좋긴함.
# '통상적으로'