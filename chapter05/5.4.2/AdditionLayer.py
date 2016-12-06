#coding: utf-8

class AdditionLayer:
	def __init__(self):
		pass
	
	def forward(self, x, y):
		z = x + y
		return z
	
	def backward(self, dz):
		dx = dz * 1
		dy = dz * 1
		return dx, dy
