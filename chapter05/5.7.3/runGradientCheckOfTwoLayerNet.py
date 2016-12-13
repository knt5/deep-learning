#coding: utf-8

import numpy as np
import sys
sys.path.append('../../')
sys.path.append('../5.7.2/')
from data.mnist import load_mnist
from TwoLayerNet import TwoLayerNet

(xTrain, tTrain), (xTest, tTest) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(inputLayerSize=784, hiddenLayerSize=50, ouputLayerSize=10)

xBatch = xTrain[:3]
tBatch = tTrain[:3]

numericalGradient = network.getNumericalGradient(xBatch, tBatch)
backpropagationGradient = network.getGradient(xBatch, tBatch)

for key in numericalGradient.keys():
	diff = np.average( np.abs(backpropagationGradient[key] - numericalGradient[key]) )
	print(key + ":" + str(diff))
