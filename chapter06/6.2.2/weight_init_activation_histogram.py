# coding: utf-8

import sys
sys.path.append('../../')
import numpy as np
import matplotlib.pyplot as plt
from common.functions import sigmoid, relu, tanh

x = np.random.randn(1000, 100)  # 1000 input data
nodeNum = 100  # node num in hidden layer
hiddenLayerSize = 5
activations = {}  # activation results

for i in range(hiddenLayerSize):
	if i != 0:
		x = activations[i-1]
	
	# initial weight (6.2.2)
	#w = np.random.randn(nodeNum, nodeNum) * 1
	#w = np.random.randn(nodeNum, nodeNum) * 0.01
	w = np.random.randn(nodeNum, nodeNum) * np.sqrt(1.0 / nodeNum)  # Xavier for S curve (sigmoid, tanh, ...)
	#w = np.random.randn(nodeNum, nodeNum) * np.sqrt(2.0 / nodeNum)  # He for ReLU
	
	a = np.dot(x, w)
	
	# activation functions (6.2.3)
	#z = sigmoid(a)
	z = tanh(a)
	#z = relu(a)
	
	activations[i] = z

# draw histgram
for i, a in activations.items():
	plt.subplot(1, len(activations), i+1)
	plt.title(str(i+1) + "-layer")
	if i != 0: plt.yticks([], [])
	#plt.xlim(0.1, 1)
	#plt.ylim(0, 7000)
	#plt.hist(a.flatten(), 30, range=(0,1))
	plt.hist(a.flatten(), 30, range=(-1,1))
plt.show()
