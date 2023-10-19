import cv2
import numpy as np
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
img1 = cv2.imread("pik.jpg")
img2 = img1.copy()
img1_r = img1.copy()
img1_g = img1.copy()
img1_b = img1.copy()
img1[:,:,0] = img1_b[:,:,2]
img1[:,:,1] = img1_g[:,:,1]
img1[:,:,2] = img1_r[:,:,0]
plt.subplot(2, 4, 1).imshow(img2, cmap=None) 
plt.title("BGR")
plt.subplot(2, 4, 2).imshow(img2[:,:,0], cmap="gray")  
plt.title("B")
plt.subplot(2, 4, 3).imshow(img2[:,:,1], cmap="gray")  
plt.title("G")
plt.subplot(2, 4, 4).imshow(img2[:,:,2], cmap="gray")  
plt.title("R")  
plt.subplot(2, 4, 5).imshow(img1, cmap=None) 
plt.title("RGB")  
plt.subplot(2,4, 6).imshow(img1[:,:,0], cmap="gray") 
plt.title("R")  
plt.subplot(2,4, 7).imshow(img1[:,:,1], cmap="gray") 
plt.title("G")  
plt.subplot(2,4, 8).imshow(img1[:,:,2], cmap="gray") 
plt.title("B")  
plt.show()
