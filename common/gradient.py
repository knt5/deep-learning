#coding: utf-8

import numpy as np

def numericalGradientWithoutBatch(f, x):
	h = 1e-4
	gradient = np.zeros_like(x)  # same shape as x with 0 array
	
	#for index in range(x.size):
	iterator = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
	while not iterator.finished:
		index = iterator.multi_index
		t = x[index]
		
		# f(x+h)
		x[index] = t + h
		y1 = f(x)
		
		# f(x-h)
		x[index] = t - h
		y2 = f(x)
		
		# add to gradient
		gradient[index] = (y1 - y2) / (2 * h)
		
		# revert
		x[index] = t
		
		iterator.iternext()
	
	return gradient

def numericalGradient(f, x):
	if x.ndim == 1:
		return numericalGradientWithoutBatch(f, x)
	else:
		gradient = np.zeros_like(x)
		for index, t in enumerate(x):
			gradient[index] = numericalGradientWithoutBatch(f, t)
		return gradient
