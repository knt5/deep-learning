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

class Affine:
	def __init__(self, w, b):
		self.w = w
		self.b = b
		self.x = None
		self.xShape = None
		
		# derivation
		self.dw = None
		self.db = None
	
	def forward(self, x):
		self.xShape = x.shape
		x = x.reshape(x.shape[0], -1)
		
		self.x = x
		y = np.dot(self.x, self.w) + self.b
		return y
	
	def backward(self, dy):
		dx = np.dot(dy, self.w.T)
		self.dw = np.dot(self.x.T, dy)
		self.db = np.sum(dy, axis=0)
		dx = dx.reshape(*self.xShape)  # Revert shape
		return dx

class SoftmaxWithLoss:
	def __init__(self):
		self.loss = None
		self.y = None  # softmax output
		self.t = None  # teacher data (one-hot-vector)
	
	def forward(self, x, t):
		self.t = t
		self.y = softmax(x)
		self.loss = crossEntropyError(self.y, self.t)
		return self.loss
	
	def backward(self, dout=1):
		batchSize = self.t.shape[0]
		if self.t.size == self.y.size:  # t is one-hot-vector
			dx = (self.y - self.t) / batchSize
		else:                           # or not
			dx = self.y.copy()
			dx[np.arange(batchSize), self.t] -= 1
			dx = dx / batchSize
		
		return dx
