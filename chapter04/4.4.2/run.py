#coding: utf-8

import sys
import numpy as np
from SimpleNet import SimpleNet
from common.gradient import numericalGradient

net = SimpleNet()
print(net.w)

x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)
print(np.argmax(p))  # index of max number

t = np.array([0, 0, 1])
loss = net.loss(x, t)
print(loss)

#def f(w):
#	return net.loss(x, t)
#dw = numericalGradient(f, net.w)
dw = numericalGradient(lambda w: net.loss(x, t), net.w)
print(dw)
