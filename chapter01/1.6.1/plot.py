#coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

# Data
x = np.arange(0, 60, 0.1)
y = np.sin(x)

# Plot
plt.plot(x, y)
plt.show()
