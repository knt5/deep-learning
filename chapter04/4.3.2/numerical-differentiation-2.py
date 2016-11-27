#coding: utf-8

import sys
import numpy as np
import matplotlib.pylab as plt
sys.path.append('../../')
from common.functions import numericalDifferentiation

def f(x):
	return 0.01 * x ** 2 + 0.1 * x

def df(x):
	return 0.02 * x + 0.1

def analyticLine(f, x):
	a = df(x)
	b = f(x) - a * x
	return lambda t: a * t + b

def numLine(f, x):
	a = numericalDifferentiation(f, x)
	b = f(x) - a * x
	return lambda t: a * t + b

x = np.arange(0.0, 20.0, 0.1)
#x = np.arange(0.0, 100.0, 0.1)

plt.xlabel('x')
plt.ylabel('f(x)')
plt.plot(x, f(x))
plt.plot(x, analyticLine(f, 5)(x))
plt.plot(x, analyticLine(f, 10)(x))
plt.plot(x, numLine(f, 5)(x), linestyle='--')
plt.plot(x, numLine(f, 10)(x), linestyle='--')
#for i in range(1, 100):
#	plt.plot(x, numLine(f, i)(x))
plt.show()
