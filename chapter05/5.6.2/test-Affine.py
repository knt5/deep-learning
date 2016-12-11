# coding: utf-8

import numpy as np
import sys
sys.path.append('../../')
from common.layers import Affine

x = np.array([[1, 2, 3], [4, 5, 6]])
w = np.array([[1, 2], [3, 4], [5, 6]])
b = np.array([1, 2])
print('--------------')
print(x)
print(w)
print(b)

affine = Affine(w, b)

#---------------------------------------
# forward
print('--------------')
y = affine.forward(x)
print(y)

#---------------------------------------
# backward
dy = np.array([[5, 5], [5, 5]])
dx = affine.backward(dy)
print(dx)
