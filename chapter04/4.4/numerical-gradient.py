#coding: utf-8

import sys
import matplotlib.pylab as plt
import numpy as np
sys.path.append('../../')
from common.gradient import numericalGradient

def f(x):
	return np.sum(x ** 2)

print(numericalGradient(f, np.array([3.0, 4.0])))
print(numericalGradient(f, np.array([0.0, 2.0])))
print(numericalGradient(f, np.array([3.0, 0.0])))
print(numericalGradient(f, np.array([99.0, 100.0, 5.0, 300.0])))
