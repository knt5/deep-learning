#coding: utf-8

import sys
sys.path.append('../5.4.1/')
from AdditionLayer import AdditionLayer
from MultiplicationLayer import MultiplicationLayer

#------------------------------
# input 🍎 🍊 💰
appleUnitPrice = 100
appleNum = 2
mikanUnitPrice = 150
mikanNum = 3
tax = 1.1

# layers
appleLayer = MultiplicationLayer()
mikanLayer = MultiplicationLayer()
amountLayer = AdditionLayer()
taxLayer = MultiplicationLayer()

# forward
applePrice = appleLayer.forward(appleUnitPrice, appleNum)
mikanPrice = mikanLayer.forward(mikanUnitPrice, mikanNum)
amount = amountLayer.forward(applePrice, mikanPrice)
price = taxLayer.forward(amount, tax)
print(price)

#------------------------------
# backward
dPrice = 1
dAmount, dTax = taxLayer.backward(dPrice)
dApplePrice, dMikanPrice = amountLayer.backward(dAmount)
dAppleUnitPrice, dAppleNum = appleLayer.backward(dApplePrice)
dMikanUnitPrice, dMikanNum = mikanLayer.backward(dMikanPrice)
print(dAppleUnitPrice, dAppleNum, dMikanUnitPrice, dMikanNum, dTax)
