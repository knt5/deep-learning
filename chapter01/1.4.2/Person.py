#coding: utf-8

class Person:
	def __init__(self, name):
		self.name = name
		print('Initialized: ' + self.name)
	
	def say(self): 		print('Say Yo! ' + self.name)
