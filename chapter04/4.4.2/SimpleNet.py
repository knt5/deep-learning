#coding: utf-8

import sys
import numpy as np
sys.path.append('../../')
from common.functions import softmax, crossEntropyError
#from common.gradient import numericalGradient

class SimpleNet:
	def __init__(self):
		self.w = np.random.randn(2, 3)  # 2x3 matrix
	
	def predict(self, x):
		return np.dot(x, self.w)
	
	def loss(self, x, t):
		z = self.predict(x)
		y = softmax(z)
		loss = crossEntropyError(y, t)
		
		return loss
