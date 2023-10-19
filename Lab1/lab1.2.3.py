import cv2
import numpy as np
from matplotlib import pyplot as plt
image1 = plt.imread("pik.jpg")
image_mask = np.zeros((image1.shape[0],image1.shape[1]),dtype=np.uint8)
white_mask = np.zeros((200, 400), dtype=np.uint8)
white_mask.fill(255) 
x_position = 200
y_position = 50
image_mask[y_position:y_position+200, x_position:x_position+400] = white_mask
result = cv2.bitwise_and(image1,image1,mask=image_mask)
plt.subplot(1,3,1).imshow(image1)
plt.title("Original")
plt.subplot(1,3,2).imshow(image_mask,cmap="gray")
plt.title("Image Mask")
plt.subplot(1,3,3).imshow(result)
plt.title("Result")
plt.show()

