#coding: utf-8

import numpy as np

# sigmoid function
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

# identify function
def identify(x):
	return x

# input layer
x = np.array([1.0, 0.5])                           # input
w1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])  # weight
b1 = np.array([0.1, 0.2, 0.3])                     # bias

# layer 1
a1 = np.dot(x, w1) + b1
z1 = sigmoid(a1)

print(a1)
print(z1)

# layer 2
w2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
b2 = np.array([0.1, 0.2])
a2 = np.dot(z1, w2) + b2
z2 = sigmoid(a2)

print(a2)
print(z2)

# output layer
w3 = np.array([[0.1, 0.3], [0.2, 0.4]])
b3 = np.array([0.1, 0.2])
a3 = np.dot(z2, w3) + b3
y = identify(a3)

print(y)
