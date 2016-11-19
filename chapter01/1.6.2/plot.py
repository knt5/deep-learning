#coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

# Data
x = np.arange(0, 12, 0.1)
y1 = np.sin(x)
y2 = np.cos(x)

# Plot
plt.plot(x, y1, label='sin')
plt.plot(x, y2, label='cos', linestyle='--')
plt.xlabel('xxxxx')
plt.ylabel('yyyyy')
plt.title('😄😄😄')
plt.legend()
plt.show()
