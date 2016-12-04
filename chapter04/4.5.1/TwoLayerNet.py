#coding: utf-8

import sys
import numpy as np
sys.path.append('../../')
from common.functions import sigmoid, softmax, crossEntropyError
from common.gradient import numericalGradient

class TwoLayerNet:
	def __init__(self, inputLayerSize, hiddenLayerSize, ouputLayerSize, weightFactor = 0.01):
		self.params = {}
		self.params['w1'] = weightFactor * np.random.randn(inputLayerSize, hiddenLayerSize)
		self.params['b1'] = np.zeros(hiddenLayerSize)
		self.params['w2'] = weightFactor * np.random.randn(hiddenLayerSize, ouputLayerSize)
		self.params['b2'] = np.zeros(ouputLayerSize)
	
	def predict(self, x):
		w1, w2 = self.params['w1'], self.params['w2']
		b1, b2 = self.params['b1'], self.params['b2']
		
		a1 = np.dot(x, w1) + b1
		z1 = sigmoid(a1)
		a2 = np.dot(z1, w2) + b2
		y = softmax(a2)
		
		return y
	
	# x:input, t:teacher
	def getLoss(self, x, t):
		y = self.predict(x)
		return crossEntropyError(y, t)
	
	# x:input, t:teacher
	def getAccuracy(self, x, t):
		y = self.predict(x)
		y = np.argmax(y, axis=1)
		t = np.argmax(t, axis=1)
		
		batchSize = float(x.shape[0])
		accuracy = np.sum(y == t) / batchSize
		return accuracy
	
	# x:input, t:teacher
	def getNumericalGradient(self, x, t):
		wLossFunc = lambda w: self.getLoss(x, t)
		
		gradients = {}
		gradients['w1'] = numericalGradient(wLossFunc, self.params['w1'])
		gradients['b1'] = numericalGradient(wLossFunc, self.params['b1'])
		gradients['w2'] = numericalGradient(wLossFunc, self.params['w2'])
		gradients['b2'] = numericalGradient(wLossFunc, self.params['b2'])
		return gradients
