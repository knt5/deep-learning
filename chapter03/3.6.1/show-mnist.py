#coding: utf-8

import sys, os
import numpy as np
from PIL import Image
sys.path.append('../../')
from data.mnist import load_mnist

def showImage(image):
	pilImage = Image.fromarray(np.uint8(image))
	pilImage.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

image = x_train[0]
label = t_train[0]
print(label)

print(image.shape)
image = image.reshape(28, 28)
print(image.shape)

showImage(image)
