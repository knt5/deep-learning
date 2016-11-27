#coding: utf-8

import sys
sys.path.append('../../')
from common.functions import numericalDifferentiation

def f(x):
	return x ** 2

print(numericalDifferentiation(f, 1))
print(numericalDifferentiation(f, 2))
print(numericalDifferentiation(f, 3))
