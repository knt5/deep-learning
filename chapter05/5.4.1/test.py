#coding: utf-8

from MultiplicationLayer import MultiplicationLayer

#------------------------------
# input
appleUnitPrice = 100
appleNum = 2
tax = 1.1

# layers
appleLayer = MultiplicationLayer()
taxLayer = MultiplicationLayer()

# forward
applePrice = appleLayer.forward(appleUnitPrice, appleNum)
price = taxLayer.forward(applePrice, tax)
print(price)

#------------------------------
# backward
dPrice = 1
dApplePrice, dTax = taxLayer.backward(dPrice)
dAppleUnitPrice, dAppleNum = appleLayer.backward(dApplePrice)
print(dAppleUnitPrice, dAppleNum, dTax)
