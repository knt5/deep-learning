#coding: utf-8

import numpy as np
import matplotlib.pylab as plt

def relu(x):
	return np.maximum(0, x)

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def step(x):
	return np.array(x > 0, dtype=np.int)

x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)
y2 = sigmoid(x) * 5
y3 = step(x) * 5

plt.plot(x, y, label='ReLU')
plt.plot(x, y2, linestyle='--', label='Sigmoid')
plt.plot(x, y3, linestyle='--', label='Step')
plt.ylim(-1.0, 7.0)  # y axis range
plt.legend()
plt.show()
