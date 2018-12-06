# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 15:49:30 2018

@author: 599798
"""

# GrayScale Image Convertor
# https://extr3metech.wordpress.com
 
import cv2
image = cv2.imread('test1.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite('gray_image.png',gray_image)
cv2.imshow('color_image',image)
cv2.imshow('gray_image',gray_image) 
cv2.waitKey(0)                 # Waits forever for user to press any key
cv2.destroyAllWindows()        # Closes displayed windows
 
#End of Code