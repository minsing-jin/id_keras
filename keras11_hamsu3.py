import numpy as np
from numpy.core.fromnumeric import transpose

# 1
x = np.array(range(100))
y = np.array([range(711,811), range(100)])
print(x.shape) # (5, 100)
print(y.shape) # (2, 100)

x = np.transpose(x)
y = np.transpose(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=66
)

print(x_train.shape)
print(y_train.shape)