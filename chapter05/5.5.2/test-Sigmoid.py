# coding: utf-8

import numpy as np
import sys
sys.path.append('../../')
from common.layers import Sigmoid

sigmoid = Sigmoid()

#---------------------------------------
# forward
x = np.array([[1.0, -0.5], [-2.0, 3.0]])
print(x)
y = sigmoid.forward(x)
print(y)

#---------------------------------------
# backward
dy = np.array([[5, 5], [5, 5]])
dx = sigmoid.backward(dy)
print(dx)
