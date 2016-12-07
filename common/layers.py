# coding: utf-8

import numpy as np
from common.functions import sigmoid, crossEntropyError

class Relu:
	def __init__(self):
		self.mask = None
	
	def forward(self, x):
		self.mask = (x <= 0)
		y = x.copy()
		y[self.mask] = 0
		return y
	
	def backward(self, dy):
		dy[self.mask] = 0
		dx = dy
		return dx
