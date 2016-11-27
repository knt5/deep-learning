#coding: utf-8

import sys
import numpy as np
sys.path.append('../../')
from common.functions import meanSquaredError

t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])

# 2 is the highest
y = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
print(meanSquaredError(y, t))

# 7 is the highest
y = np.array([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0])
print(meanSquaredError(y, t))
