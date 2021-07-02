from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Convolution2D, Flatten
from tensorflow.python.keras.layers.convolutional import Conv2D

# 이미지 자르는 것은 convolution layer에서 함 -> 2d(그림,그래프) 1d(선) 다 있음 -> 이미지의 차원
model = Sequential()
model.add(Conv2D(filters=10, kernel_size=(2,2), strides=1, input_shape = (5,5,1)))
# 이미지를 수치화하는데 dense layer로 조각조각내서 특성값을 구할것임
model.add(Conv2D(5, (2,2), padding='same'))   #제일 앞에있는 수치는 filter로 지정, stride는 명시를 하지 않으면 디폴트는 1이다, padding은 shape맞출때 많이 씀, 그리고 이미지 슬라이스해서 stride로 맨끝 이미지 손실되는부분 보완?
# 바로 밑에는 굳이 filters것과 같은 이름들을 명시하지 않아도됌! 
model.add(Flatten())
model.add(Dense(1))  #dense를 2개로 할거면 sigmoid나 softmax로 activation파라미터를 넣어도 괜찮

model.summary()


# 왜 자꾸 input_shape가 잘못된 argument라고 나오지
# output하고 filter수를 어떻게 들어가는거임? output shape의 변화를 모르겠음.
# 5,5,3 데이터를 가로로 쭉 이어붙이면~~~
# 이미지는 보통 분류모델많이씀
# 최종레이어는 대부분 Dense임 
# flatten은 평평하게 만듬, 잘라서 가로로 쭉 이어줌
# 왜 자꾸 input shape는 정의되지 않을까--> input_shape에 =으로 하기 
# 이미지 문제를 DNN문제로 푸는것들도 있음. 이미지를 Dense로 할 수 있음.
# 한마디로 이미지 역시 숫자고 특정있는 숫자를 잘라서 나중에 최종 비교할대 sigmoid로 하거나 softmax로하거나 해서 남자인지 여자인지 구분

# 과제
# 서머리에서 파라미터가 왜 50이고 205고 토탈 파라미터는 336인지 이해했다는것 확인
# 궁금한건 이메일로 질문할 것
# 여기까지 한걸 이해해놓을 것










# 선배 코드 및 정리
'''
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(filters=10, kernel_size = (2,2), strides = 1, input_shape = (5, 5, 1)))
# 연산하면 (5, 5, 1) 에서 (4, 4, 10 ) 로 됨. 
# padding = 'same'으로 하면 shape 그대로 가져감. 장점 : 손실방지
model.add(Conv2D(5, (2,2), padding = 'same')) 
#연산하면 4, 4, 10 에서-> 4, 4, 5 가 됨. 4, 4 그대로 가져가고.. 뒤에 filter값 

# 젤 앞에있는 수치 필터(노드갯수_아웃풋)로 인식
# 다음수치 커널사이즈, stride는 디폴트1
model.add(Flatten())

model.add(Dense(1))


model.summary()

# 레이어에서 이미지를 머신이 알 수 있게 수치화 하는 작업 시행
# 특성값을 찾기 위해 조각내기. 커널사이즈 = 자르는 조각 크기 . . 
# stride = 몇번씩 이동할건지, filter = (아웃풋)노드 갯수
'''