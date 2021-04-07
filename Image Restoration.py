#!/usr/bin/env python
# coding: utf-8

# In[9]:


# import packages for picture loading, data processing and graph plotting
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt,exp
import cv2


# Define distance function to locate the points in the frequency domain
def distance(point1,point2):
    return sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

# Define Low Pass filter function
def idealFilterLP(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            if distance((y,x),center) < D0:
                base[y,x] = 1
    return base
# Define High Pass filter function
def idealFilterHP(D0,imgShape):
    base = np.ones(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            if distance((y,x),center) < D0:
                base[y,x] = 0
    return base
    
# Begin FFT with low-pass filter
plt.figure(figsize=(7*5, 5*5), constrained_layout=False)
# Load the original picture
img = cv2.imread("poppy.jpg", 0)
plt.subplot(131), plt.imshow(img, "gray"), plt.title("Original Image")
# Perform FFT 
original = np.fft.fft2(img)
plt.subplot(132), plt.imshow(np.log(1+np.abs(original)), "gray"), plt.title("Spectrum")
# Perform centralization
center = np.fft.fftshift(original)
plt.subplot(133), plt.imshow(np.log(1+np.abs(center)), "gray"), plt.title("Centered Spectrum")

# plot the graph
plt.gcf()
# save the modified image
plt.savefig('LP_poppy1.jpg')

plt.figure(figsize=(7*5, 5*5), constrained_layout=False)
# Perform centralization with Low Pass filter
LowPassCenter = center * idealFilterLP(50,img.shape)
plt.subplot(231), plt.imshow(np.log(1+np.abs(LowPassCenter)), "gray"), 
plt.title("Centered Spectrum multiply Low Pass Filter")
# Perform decentralization
LowPass = np.fft.ifftshift(LowPassCenter)
plt.subplot(232), plt.imshow(np.log(1+np.abs(LowPass)), "gray"), plt.title("Decentralize")
# Perfrom inverse FFT to obtain the modified image
inverse_LowPass = np.fft.ifft2(LowPass)
plt.subplot(233), plt.imshow(np.abs(inverse_LowPass), "gray"), plt.title("Processed Image")

# plot the graph
plt.gcf()
# save the modified image
plt.savefig('LP_poppy2.jpg')

# Begin FFT with High-pass filter
plt.figure(figsize=(7*5, 5*5), constrained_layout=False)
# Load the original image
img = cv2.imread("poppy.jpg", 0)
plt.subplot(131), plt.imshow(img, "gray"), plt.title("Original Image")
# Perform FFT 
original = np.fft.fft2(img)
plt.subplot(132), plt.imshow(np.log(1+np.abs(original)), "gray"), plt.title("Spectrum")
# Perform centralization
center = np.fft.fftshift(original)
plt.subplot(133), plt.imshow(np.log(1+np.abs(center)), "gray"), plt.title("Centered Spectrum")

# plot the graph
plt.gcf()
# save the modified image
plt.savefig('HP_poppy1.jpg')

# Perform centralization with Low Pass filter
plt.figure(figsize=(7*5, 5*5), constrained_layout=False)
HighPassCenter = center * idealFilterHP(50,img.shape)
plt.subplot(231), plt.imshow(np.log(1+np.abs(HighPassCenter)), "gray"), 
plt.title("Centered Spectrum multiply Low Pass Filter")
# Perform decentralization
HighPass = np.fft.ifftshift(HighPassCenter)
plt.subplot(232), plt.imshow(np.log(1+np.abs(HighPass)), "gray"), plt.title("Decentralize")
# Perfrom inverse FFT to obtain the modified image
inverse_HighPass = np.fft.ifft2(HighPass)
plt.subplot(233), plt.imshow(np.abs(inverse_HighPass), "gray"), plt.title("Processed Image")

# plot the graph
plt.gcf()
# save the modified image
plt.savefig('HP_poppy2.jpg')

