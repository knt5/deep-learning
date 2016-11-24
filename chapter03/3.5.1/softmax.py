#coding: utf-8

import numpy as np

# softmax function
def softmax(a):
	expA = np.exp(a)
	sumExpA = np.sum(expA)
	return expA / sumExpA

a = np.array([0.3, 2.9, 4.0])
print(softmax(a))
