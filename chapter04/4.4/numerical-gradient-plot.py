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

x0 = np.arange(-2, 2.5, 0.25)
x1 = np.arange(-2, 2.5, 0.25)
x, y = np.meshgrid(x0, x1)
x = x.flatten()
y = y.flatten()

gradient = numericalGradient(f, np.array([x, y]))
print(gradient)

plt.figure()
plt.quiver(x, y, -gradient[0], -gradient[1], angles="xy", color="#ff6666", headwidth=8, scale=40)
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.legend()
plt.draw()
plt.show()
