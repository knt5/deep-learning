#coding: utf-8

import sys
import pickle
import numpy as np
from PIL import Image
sys.path.append('../../')
from data.mnist import load_mnist
from common.functions import sigmoid, softmax

def getData():
	(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
	return x_test, t_test

def initNetwork():
	# sample_weight.pkl
	# -> original: https://github.com/oreilly-japan/deep-learning-from-scratch/tree/master/ch03
	weightFilePath = open('sample_weight.pkl', 'rb')
	network = pickle.load(weightFilePath)
	return network

def predict(network, x):
	w1, w2, w3 = network['W1'], network['W2'], network['W3']
	b1, b2, b3 = network['b1'], network['b2'], network['b3']
	
	a1 = np.dot(x, w1) + b1
	z1 = sigmoid(a1)
	a2 = np.dot(z1, w2) + b2
	z2 = sigmoid(a2)
	a3 = np.dot(z2, w3) + b3
	y = softmax(a3)
	
	return y

x, t = getData()
network = initNetwork()
#print(network)

correctCount = 0
for i in range(len(x)):
	#print(x[i])
	y = predict(network, x[i])
	#print(y)
	index = np.argmax(y)  # the biggest element's index
	if index == t[i]:
		correctCount += 1

print(
	'correctCount: ' + str(correctCount) + '\n' +
	'len(x): ' + str(len(x)) + '\n' +
	'Accuracy: ' + str(float(correctCount) / len(x))
)
