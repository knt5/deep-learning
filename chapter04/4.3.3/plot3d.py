#coding: utf-8

import sys
import numpy as np
import matplotlib.pylab as plt
sys.path.append('../../')
from common.functions import numericalDifferentiation
from mpl_toolkits.mplot3d import Axes3D

#def f(x):
#	return np.sum(x ** 2)

def f(x, y):
	return x ** 2 + y ** 2

x = np.arange(-3.0, 3.0, 0.25)
y = np.arange(-3.0, 3.0, 0.25)
#x = np.arange(-3.0, 3.0, 0.1)
#y = np.arange(-3.0, 3.0, 0.1)
x, y = np.meshgrid(x, y)
z = f(x, y)

figure = plt.figure()
axes = Axes3D(figure)
axes.plot_wireframe(x, y, z)
plt.show()
