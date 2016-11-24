#coding: utf-8

import numpy as np

# softmax function
def softmax(a):
	c = np.max(a)
	expA = np.exp(a - c)  # a - c : protected from overflow
	sumExpA = np.sum(expA)
	return expA / sumExpA

a = np.array([0.3, 2.9, 4.0])
print(softmax(a))

a = np.array([1010, 1000, 990])
print(softmax(a))
