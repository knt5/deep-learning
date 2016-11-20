#coding: utf-8

import numpy as np

def AND(x1, x2):
	x = np.array([x1, x2])
	w = np.array([0.5, 0.5])
	bias = -0.7
	tmp = np.sum(w * x) + bias
	
	if tmp <= 0:
		return 0
	elif tmp > 0:
		return 1

def NAND(x1, x2):
	x = np.array([x1, x2])
	w = np.array([-0.5, -0.5])  # diff is weight and bias ONLY!
	bias = 0.7                  #
	tmp = np.sum(w * x) + bias
	
	if tmp <= 0:
		return 0
	elif tmp > 0:
		return 1

def OR(x1, x2):
	x = np.array([x1, x2])
	w = np.array([0.5, 0.5])  # diff is weight and bias ONLY!
	bias = -0.2               #
	tmp = np.sum(w * x) + bias
	
	if tmp <= 0:
		return 0
	elif tmp > 0:
		return 1

# Multi-layered perceptron
def XOR(x1, x2):
	s1 = NAND(x1 ,x2)
	s2 = OR(x1, x2)
	y = AND(s1, s2)
	return y

print('----------------- AND')
print(AND(0, 0))
print(AND(0, 1))
print(AND(1, 0))
print(AND(1, 1))

print('----------------- NAND')
print(NAND(0, 0))
print(NAND(0, 1))
print(NAND(1, 0))
print(NAND(1, 1))

print('----------------- OR')
print(OR(0, 0))
print(OR(0, 1))
print(OR(1, 0))
print(OR(1, 1))

print('----------------- XOR')
print(XOR(0, 0))
print(XOR(0, 1))
print(XOR(1, 0))
print(XOR(1, 1))
