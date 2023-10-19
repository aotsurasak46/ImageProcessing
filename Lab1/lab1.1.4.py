import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True
x = cv2.imread("fuji.jpg")
x = cv2.resize(x, (200, 200))
gray_image = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
xx, yy = np.mgrid[0:gray_image.shape[0], 0:gray_image.shape[1]]
fig = plt.figure()
ax = fig.add_subplot(121, projection="3d")
ax.plot_surface(xx, yy, gray_image, rstride=1, cstride=1, cmap='gray')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Intensity')
ax.set_zlim(0, 255)
ax.view_init(elev=70, azim=45)
ax.set_title("3d")
plt.subplot(1,2,2).imshow(gray_image,cmap="gray")
plt.title("original")
plt.show()
