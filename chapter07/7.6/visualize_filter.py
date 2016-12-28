# coding: utf-8

import sys
sys.path.append('../7.5/')
import numpy as np
import matplotlib.pyplot as plt
from SimpleCNN import SimpleCNN

def filter_show(filters, nx=8, margin=3, scale=10):
	"""
	c.f. https://gist.github.com/aidiary/07d530d5e08011832b12#file-draw_weight-py
	"""
	FN, C, FH, FW = filters.shape
	ny = int(np.ceil(FN / nx))
	
	fig = plt.figure()
	fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
	
	for i in range(FN):
		ax = fig.add_subplot(ny, nx, i+1, xticks=[], yticks=[])
		ax.imshow(filters[i, 0], cmap=plt.cm.gray_r, interpolation='nearest')
	plt.show()

network = SimpleCNN()

# ランダム初期化後の重み
filter_show(network.params['w1'])

# 学習後の重み
network.load_params('../7.5/params.pkl')
filter_show(network.params['w1'])
