#coding: utf-8

import numpy as np

def identify(x):
	return x

def step(x):
	return np.array(x > 0, dtype=np.int)

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def relu(x):
	return np.maximum(0, x)

def softmax(a):
	c = np.max(a)
	expA = np.exp(a - c)  # a - c : protected from overflow
	sumExpA = np.sum(expA)
	return expA / sumExpA
