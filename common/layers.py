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

class Sigmoid:
	def __init__(self):
		self.y = None
	
	def forward(self, x):
		y = sigmoid(x)
		self.y = y
		return y
	
	def backward(self, dy):
		dx = dy * (1.0 - self.y) * self.y
		return dx
