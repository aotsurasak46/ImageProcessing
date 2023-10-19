import cv2
import numpy as np
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
x = plt.imread("pik.jpg")
xr = x[:,:,0]
x_transpose = xr.transpose()
x_moveaxis = np.moveaxis(x_transpose,0,1)
x_reshape = np.reshape(x,(x.shape[2],x.shape[0],x.shape[1]))
print(x.shape)
print(x_transpose.shape)
print(x_moveaxis.shape)
print(x_reshape.shape)
plt.subplot(1, 4, 1).imshow(xr, cmap="gray") 
plt.title("R_original")
plt.subplot(1, 4, 2).imshow(x_transpose, cmap="gray")
plt.title("R_transpose")
plt.subplot(1, 4, 3).imshow(x_moveaxis, cmap="gray") 
plt.title("R_moveaxis")
plt.subplot(1, 4, 4).imshow(x_reshape[0,:,:], cmap="gray") 
plt.title("R_reshape")
plt.show()

