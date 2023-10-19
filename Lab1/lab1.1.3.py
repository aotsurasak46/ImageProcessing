import cv2
import numpy as np
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
x = cv2.imread("fuji.jpg")
gray_image = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
gray_image = cv2.resize(gray_image,(200,200))
quantized_image = np.floor((gray_image/255)*4)
plt.subplot(1, 2, 1).imshow(gray_image, cmap="gray")
plt.title("original")
plt.subplot(1, 2, 2).imshow(quantized_image, cmap="gray")
plt.title("quantized")
plt.show()

