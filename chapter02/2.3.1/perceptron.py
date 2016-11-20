#coding: utf-8

def AND(x1, x2):
	w1, w2, theta = 0.5, 0.5, 0.7
	tmp = w1 * x1 + w2 * x2
	if tmp <= theta:
		return 0
	elif tmp > theta:
		return 1

print(AND(0, 0))
print(AND(0, 1))
print(AND(1, 0))
print(AND(1, 1))
