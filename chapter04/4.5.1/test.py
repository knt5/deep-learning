#coding: utf-8

import numpy as np
from TwoLayerNet import TwoLayerNet

net = TwoLayerNet(784, 20, 10)
print(net.params['w1'].shape)
print(net.params['b1'].shape)
print(net.params['w2'].shape)
print(net.params['b2'].shape)

# predict with dummy input data (x20)
x = np.random.rand(20, 784)
y = net.predict(x)
#print(y)

# gradient with dummy
t = np.random.rand(20, 10)  # dummy "answer" labels (x20)
gradient = net.getNumericalGradient(x, t)
print('----------------------- w1')
print(gradient['w1'])
print('----------------------- b1')
print(gradient['b1'])
print('----------------------- w2')
print(gradient['w2'])
print('----------------------- b2')
print(gradient['b2'])
print('----------------------- shape')
print(gradient['w1'].shape)
print(gradient['b1'].shape)
print(gradient['w2'].shape)
print(gradient['b2'].shape)
