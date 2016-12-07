# coding: utf-8

import numpy as np
import sys
sys.path.append('../../')
from common.layers import Relu

relu = Relu()

#---------------------------------------
# forward
x = np.array([[1.0, -0.5], [-2.0, 3.0]])
print(x)
y = relu.forward(x)
print(y)

#---------------------------------------
# backward
dy = np.array([[5, 5], [5, 5]])
dx = relu.backward(dy)
print(dx)
