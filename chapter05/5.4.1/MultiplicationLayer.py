#coding: utf-8

class MultiplicationLayer:
	def __init__(self):
		self.x = None
		self.y = None
	
	def forward(self, x, y):
		self.x = x
		self.y = y
		z = x * y
		return z
	
	def backward(self, dz):
		dx = dz * self.y
		dy = dz * self.x
		return dx, dy
