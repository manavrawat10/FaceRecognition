# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 11:48:23 2018

@author: 599798
"""

import numpy as np
import cv2
print(cv2.__version__)
# Load an color image in grayscale
img = cv2.imread("D:\\Python\\Face\\try.jpg",0)
print(img)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()