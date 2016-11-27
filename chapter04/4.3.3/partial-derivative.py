#coding: utf-8

import sys
import numpy as np
sys.path.append('../../')
from common.functions import numericalDifferentiation

def f(x):
	return np.sum(x ** 2)

def f0(x0):
	return x0 ** 2 + 4.0 ** 2

def f1(x1):
	return 3.0 ** 2 + x1 ** 2

print(numericalDifferentiation(f0, 3.0))
print(numericalDifferentiation(f1, 4.0))
