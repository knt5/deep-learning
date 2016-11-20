#coding: utf-8

import numpy as np
import matplotlib.pylab as plt

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def step(x):
	return np.array(x > 0, dtype=np.int)

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
y2 = step(x)

plt.plot(x, y)
plt.plot(x, y2, linestyle='--')
plt.ylim(-0.1, 1.1)  # y axis range
plt.show()
