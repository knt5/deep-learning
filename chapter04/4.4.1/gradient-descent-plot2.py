#coding: utf-8

import sys
import matplotlib.pylab as plt
import numpy as np
sys.path.append('../../')
from common.gradient import numericalGradient

def f(x):
	return np.sum(x ** 2)

def line(f, x):
	a = numericalGradient(f, x)
	y = f(x) - a * x
	return lambda t: a * t + y

def gradientDescent(f, initialX, learningRate = 0.01, stepNum = 100):
	x = initialX
	history = []
	
	for i in range(stepNum):
		history.append(x.copy())
		
		gradient = numericalGradient(f, x)
		x -= learningRate * gradient
	
	return x, np.array(history)

# gradient
x0 = np.arange(-4.0, 4.0, 0.25)
x1 = np.arange(-4.0, 4.0, 0.25)
x, y = np.meshgrid(x0, x1)
x = x.flatten()
y = y.flatten()
gradient = numericalGradient(f, np.array([x, y]))
print(gradient)

# history
initialX = np.array([-3.0, 4.0])
dummy, history = gradientDescent(f, initialX, 0.1, 100)
print(dummy)
print(history)

# plot
plt.figure()
plt.quiver(x, y, -gradient[0], -gradient[1], angles="xy", color="#ff6666", headwidth=8, scale=40)
plt.plot( [-5, 5], [0,0], '--b')
plt.plot( [0,0], [-5, 5], '--b')
plt.plot(history[:,0], history[:,1], 'o')
plt.xlim([-4, 4])
plt.ylim([-4, 4])
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.legend()
plt.draw()
plt.show()
