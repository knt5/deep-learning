# coding: utf-8

import numpy as np
import sys
sys.path.append('../../')
from common.layers import SoftmaxWithLoss

softmaxWithLoss = SoftmaxWithLoss()

#---------------------------------------
x = np.array([0.3, 0.2, 0.5])
t = np.array([0, 1, 0])
out = softmaxWithLoss.forward(x, t)
print(out)
dx = softmaxWithLoss.backward(1)
print(dx)

#---------------------------------------
x = np.array([0.01, 0.99, 0.0])
t = np.array([0, 1, 0])
out = softmaxWithLoss.forward(x, t)
print(out)
dx = softmaxWithLoss.backward(1)
print(dx)
