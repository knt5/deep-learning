#coding: utf-8

import numpy as np

# sigmoid function
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

# identify function
def identify(x):
	return x

# Create network
def createNetwork():
	network = {
		'w': [
			np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]),
			np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]),
			np.array([[0.1, 0.3], [0.2, 0.4]]),
		],
		'b': [
			np.array([0.1, 0.2, 0.3]),
			np.array([0.1, 0.2]),
			np.array([0.1, 0.2]),
		]
	}
	
	return network

# Forward
def forward(network, x):
	w1, w2, w3 = network['w']
	b1, b2, b3 = network['b']
	
	a1 = np.dot(x, w1) + b1
	z1 = sigmoid(a1)
	a2 = np.dot(z1, w2) + b2
	z2 = sigmoid(a2)
	a3 = np.dot(z2, w3) + b3
	y = identify(a3)
	
	return y

# main
network = createNetwork()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)
