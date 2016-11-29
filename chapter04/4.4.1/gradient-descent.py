#coding: utf-8

import sys
#import matplotlib.pylab as plt
import numpy as np
sys.path.append('../../')
from common.gradient import numericalGradient

def f(x):
	return np.sum(x ** 2)

def gradientDescent(f, initialX, learningRate = 0.01, stepNum = 100):
	x = initialX
	
	for i in range(stepNum):
		gradient = numericalGradient(f, x)
		x -= learningRate * gradient
	
	return x

initialX = np.array([-3.0, 4.0])
print(gradientDescent(f, initialX, 0.1, 100))
