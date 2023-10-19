import tensorflow as tf
import numpy as np
import cv2
from matplotlib import pyplot as plt
from keras.models import Model
import keras as keras
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
from numpy import expand_dims

img_path = 'img/fair.jpeg'
img = image.load_img(img_path)
img = image.img_to_array(img)
print(img.shape)
img = img.reshape((1,img.shape[0],img.shape[1],img.shape[2]))
print(img.shape)
img = cv2.resize(img[0],(244,244))
print(img.shape)
imgB,imgG,imgR = cv2.split(img)
meanB = cv2.mean(imgB)
meanG = cv2.mean(imgG)
meanR = cv2.mean(imgR)
img_mean = [meanB[0],meanG[0],meanR[0]]
imgB = imgB - img_mean[0]
imgG = imgG - img_mean[1]
imgR = imgR - img_mean[2]
img = cv2.merge([imgB,imgG,imgR])
plt.imshow(img)
plt.show()