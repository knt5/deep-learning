# coding: utf-8

import numpy as np
from common.functions import sigmoid, softmax, crossEntropyError

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

class BatchNormalization:
	"""
	http://arxiv.org/abs/1502.03167
	"""
	def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
		self.gamma = gamma
		self.beta = beta
		self.momentum = momentum
		self.input_shape = None # Conv層の場合は4次元、全結合層の場合は2次元
		
		# テスト時に使用する平均と分散
		self.running_mean = running_mean
		self.running_var = running_var
		
		# backward時に使用する中間データ
		self.batch_size = None
		self.xc = None
		self.std = None
		self.dgamma = None
		self.dbeta = None
	
	def forward(self, x, train_flg=True):
		self.input_shape = x.shape
		if x.ndim != 2:
			N, C, H, W = x.shape
			x = x.reshape(N, -1)
		
		out = self.__forward(x, train_flg)
		
		return out.reshape(*self.input_shape)
			
	def __forward(self, x, train_flg):
		if self.running_mean is None:
			N, D = x.shape
			self.running_mean = np.zeros(D)
			self.running_var = np.zeros(D)
						
		if train_flg:
			mu = x.mean(axis=0)
			xc = x - mu
			var = np.mean(xc**2, axis=0)
			std = np.sqrt(var + 10e-7)
			xn = xc / std
			
			self.batch_size = x.shape[0]
			self.xc = xc
			self.xn = xn
			self.std = std
			self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
			self.running_var = self.momentum * self.running_var + (1-self.momentum) * var
		else:
			xc = x - self.running_mean
			xn = xc / ((np.sqrt(self.running_var + 10e-7)))
			
		out = self.gamma * xn + self.beta 
		return out
	
	def backward(self, dout):
		if dout.ndim != 2:
			N, C, H, W = dout.shape
			dout = dout.reshape(N, -1)
		
		dx = self.__backward(dout)
		
		dx = dx.reshape(*self.input_shape)
		return dx
	
	def __backward(self, dout):
		dbeta = dout.sum(axis=0)
		dgamma = np.sum(self.xn * dout, axis=0)
		dxn = self.gamma * dout
		dxc = dxn / self.std
		dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
		dvar = 0.5 * dstd / self.std
		dxc += (2.0 / self.batch_size) * self.xc * dvar
		dmu = np.sum(dxc, axis=0)
		dx = dxc - dmu / self.batch_size
		
		self.dgamma = dgamma
		self.dbeta = dbeta
		
		return dx
