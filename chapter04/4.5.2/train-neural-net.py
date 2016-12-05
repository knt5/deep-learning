#coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('../../')
sys.path.append('../4.5.1/')
from data.mnist import load_mnist
from TwoLayerNet import TwoLayerNet

# Load mnist data
(xTrain, tTrain), (xTest, tTest) = load_mnist(normalize=True, one_hot_label=True)

# Create a 2-layer neural network
network = TwoLayerNet(inputLayerSize=784, hiddenLayerSize=50, ouputLayerSize=10)

# Hyper parameters
stepNum = 10000
learningRate = 0.1
trainSize = xTrain.shape[0]
batchSize = 100

# Loss values in training
lossHistory = []

for i in range(stepNum):
	# Get mini batch
	batchMask = np.random.choice(trainSize, batchSize)
	xBatch = xTrain[batchMask]
	tBatch = tTrain[batchMask]
	
	# Calc gradients
	gradient = network.getGradient(xBatch, tBatch)
	
	# Update parameters
	for key in ('w1', 'b1', 'w2', 'b2'):
		network.params[key] -= learningRate * gradient[key]
	
	# Add learning history
	loss = network.getLoss(xBatch, tBatch)
	print(loss)
	lossHistory.append(loss)

# Plot learning history
x = np.arange(len(lossHistory))
plt.plot(x, lossHistory, label='loss')
plt.xlabel("iteration")
plt.ylabel("loss")
plt.show()
