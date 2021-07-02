# 과제
# R2를 음수가 아닌 0.5 이하로 만들것
# 1. 레이어는 input, output 포함해서 6개 이상
# 2. batch_size = 1
# 3. epochs = 100 이상
# 4. 히든레이어의 노드의 갯수는 10 이상 1000이하
# 5. 데이터 조작 금지

from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1. data
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15])
y_test = np.array([11,12,13,14,15])
x_pred = np.array([16,17,18])

# 2. 모델구성

# 3. 컴파일, 훈련

# 4. 평가, 예측