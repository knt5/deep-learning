#coding: utf-8

import numpy as np

def identify(x):
	return x

def step(x):
	return np.array(x > 0, dtype=np.int)

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def sigmoidGradient(x):
	return (1.0 - sigmoid(x)) * sigmoid(x)

def relu(x):
	return np.maximum(0, x)

def softmax(a):
	c = np.max(a)
	expA = np.exp(a - c)  # a - c : guard from overflow
	sumExpA = np.sum(expA)
	return expA / sumExpA

def meanSquaredError(y, t):
	return 0.5 * np.sum((y - t) ** 2)

def crossEntropyError(y, t):
	if y.ndim == 1:
		t = t.reshape(1, t.size)
		y = y.reshape(1, y.size)
	
	# Convert one-hot-vector to index
	if t.size == y.size:
		t = t.argmax(axis=1)
	
	#return - np.sum(t * np.log(y + 1e-7))  # 1e-7 : guard from -inf (np.log(0))
	batchSize = y.shape[0]
	return - np.sum(np.log(y[np.arange(batchSize), t])) / batchSize

def numericalDifferentiation(f, x):
	h = 1e-4
	return (f(x + h) - f(x - h)) / (2 * h)
