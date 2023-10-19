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
model = VGG16(weights = 'imagenet', include_top = False)
model.summary()
kernels, biases = model.layers[1].get_weights()
model.layers[1].get_config()

img_path = 'img/jems.png'
img = image.load_img(img_path, target_size=(224, 224))
img = image.img_to_array(img)
#reshape 3D(H,W,Ch) image to 4D image (sample=1,H,W,Ch)
img = expand_dims(img, axis=0)
img = preprocess_input(img)

model = Model(inputs=model.inputs, outputs=model.layers[1].output)
model.summary()
feature_map = model.predict(img)
rows = 8
columns = 8
print(feature_map.shape)

for i in range(feature_map.shape[3]):
    plt.subplot(rows,columns,i+1)
    plt.imshow(feature_map[0,:,:,i])
    plt.axis("off")
plt.show()


