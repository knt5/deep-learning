#coding: utf-8

import numpy as np
import sys
sys.path.append('../../')
from common.layers import Affine, Relu, SoftmaxWithLoss
from common.gradient import numericalGradient
from collections import OrderedDict

class TwoLayerNet:
	def __init__(self, inputLayerSize, hiddenLayerSize, ouputLayerSize, distributionScale = 0.01):
		# Initialize weight
		self.params = {}
		self.params['w1'] = distributionScale * np.random.randn(inputLayerSize, hiddenLayerSize)
		self.params['b1'] = np.zeros(hiddenLayerSize)
		self.params['w2'] = distributionScale * np.random.randn(hiddenLayerSize, ouputLayerSize)
		self.params['b2'] = np.zeros(ouputLayerSize)
		
		# Create layers
		self.layers = OrderedDict()
		self.layers['affine1'] = Affine(self.params['w1'], self.params['b1'])
		self.layers['relu1'] = Relu()
		self.layers['affine2'] = Affine(self.params['w2'], self.params['b2'])
		self.lastLayer = SoftmaxWithLoss()
	
	def predict(self, x):
		for layer in self.layers.values():
			x = layer.forward(x)
		return x
	
	def getLoss(self, x, t):
		y = self.predict(x)
		return self.lastLayer.forward(y, t)
	
	def getAccuracy(self, x, t):
		y = self.predict(x)
		y = np.argmax(y, axis=1)
		if t.ndim != 1:
			t = np.argmax(t, axis=1)
		
		accuracy = np.sum(y == t) / float(x.shape[0])
		return accuracy
	
	def getGradient(self, x, t):
		# forward
		self.getLoss(x, t)
		
		# backward
		dout = 1
		dout = self.lastLayer.backward(dout)
		layers = list(self.layers.values())
		layers.reverse()
		for layer in layers:
			dout = layer.backward(dout)
		
		gradients = {}
		gradients['w1'], gradients['b1'] = self.layers['affine1'].dw, self.layers['affine1'].db
		gradients['w2'], gradients['b2'] = self.layers['affine2'].dw, self.layers['affine2'].db
		return gradients
	
	# Numerical gradient for recalculation
	def getNumericalGradient(self, x, t):
		loss = lambda W: self.getLoss(x, t)
		gradients = {}
		gradients['w1'] = numericalGradient(loss, self.params['w1'])
		gradients['b1'] = numericalGradient(loss, self.params['b1'])
		gradients['w2'] = numericalGradient(loss, self.params['w2'])
		gradients['b2'] = numericalGradient(loss, self.params['b2'])
		return gradients
