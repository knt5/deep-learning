#coding: utf-8

import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt

# Data
x = np.arange(10, 250, 0.1)
y1 = np.sin(x / 50) * 50
y2 = np.cos(x / 50) * 50

# Image
img = imread('../../data/lena.png')

# Plot
plt.imshow(img)
plt.plot(x, y1, label='sin')
plt.plot(x, y2, label='cos', linestyle='--')
plt.title('Lena')
plt.legend()
plt.show()
