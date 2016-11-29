#coding: utf-8

import sys
import matplotlib.pylab as plt
import numpy as np
sys.path.append('../../')
from common.gradient import numericalGradient

def f(x):
	return np.sum(x ** 2)

def gradientDescent(f, initialX, learningRate = 0.01, stepNum = 100):
	x = initialX
	history = []
	
	for i in range(stepNum):
		history.append(x.copy())
		
		gradient = numericalGradient(f, x)
		x -= learningRate * gradient
	
	return x, np.array(history)

initialX = np.array([-3.0, 4.0])
x, history = gradientDescent(f, initialX, 0.1, 100)

print(x)
print(history)

plt.plot( [-5, 5], [0,0], '--b')
plt.plot( [0,0], [-5, 5], '--b')
plt.plot(history[:,0], history[:,1], 'o')
plt.xlim(-3.5, 3.5)
plt.ylim(-4.5, 4.5)
plt.xlabel("X0")
plt.ylabel("X1")
plt.show()
