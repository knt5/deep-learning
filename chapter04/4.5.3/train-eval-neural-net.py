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

# History
lossHistory = []       # Loss values in training
trainingAccuracyHistory = []  # Accuracy for training data
testAccuracyHistory = []  # Accuracy for test data

# iteration number per epoch
iterationPerEpoch = max(trainSize / batchSize, 1)

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
	#print(loss)
	lossHistory.append(loss)
	
	# Calc accuracy
	if i % iterationPerEpoch == 0:
		trainingAccuracy = network.getAccuracy(xTrain, tTrain)
		testAccuracy = network.getAccuracy(xTest, tTest)
		trainingAccuracyHistory.append(trainingAccuracy)
		testAccuracyHistory.append(testAccuracy)
		print("trainingAccuracy=" + str(trainingAccuracy) + ", testAccuracy=" + str(testAccuracy))

# Plot accuracy history
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(trainingAccuracyHistory))
plt.plot(x, trainingAccuracyHistory, label='training accuracy')
plt.plot(x, testAccuracyHistory, label='test accuracy', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
#plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
