#Array, image processing
import cv2
import numpy as np
import matplotlib.pyplot as plt
#Model Operation
from keras import Model, Input
import keras.utils as image
import tensorflow as tf
from keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, UpSampling2D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
# io
import glob
from tqdm import tqdm
import warnings;
warnings.filterwarnings('ignore')
all_images_file = glob.glob("/lustre/ai/dataset/dip/Face_mini/**/*.jpg")
from keras.utils.image_utils import img_to_array
images = []
array_images = []
for i in range(len(all_images_file)) :
  images.append(image.load_img(all_images_file[i],(100,100),interpolation="nearest"))
  array_images.append(img_to_array(images[i]))
  array_images[i]/=255
train_x,test_x = train_test_split(array_images,random_state=25,test_size=0.3)
train_x,val_x = train_test_split(train_x,random_state=13,test_size=0.2)
noise_mean = 0
noise_std = 0.5
noise_factor = 0.3
train_x_noise = train_x + (noise_factor * np.random.normal(loc=noise_mean, scale=noise_std, size=imgs.shape) )
val_x_noise = val_x + (noise_factor * np.random.normal(loc=noise_mean, scale=noise_std, size=imgs.shape) )
test_x_noise = test_x + (noise_factor * np.random.normal(loc=noise_mean, scale=noise_std, size=imgs.shape) )
plt.figure(figsize=(8, 4))
for i in range(3):
  plt.subplot(1, 3, i+1)
  plt.imshow(x_test[i])
  plt.axis('off')
plt.show()
plt.figure(figsize=(8, 4))
for i in range(3):
  plt.subplot(1, 3, i+1)
  plt.imshow(test_x_noise[i])
  plt.axis('off')
plt.show()