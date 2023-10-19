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
from keras.utils.image_utils import img_to_array
warnings.filterwarnings('ignore')
all_images_file = glob.glob("/lustre/ai/dataset/dip/Face_full_dataset/lfw/**/*.jpg")
images = []
array_images = []
for i in range(len(all_images_file)) :
  images.append(image.load_img(all_images_file[i],target_size=(80,80),interpolation="nearest"))
  array_images.append(img_to_array(images[i]))
  array_images[i]/=255

train_x,test_x = train_test_split(array_images,random_state=25,test_size=0.3)
train_x,val_x = train_test_split(train_x,random_state=13,test_size=0.2)
noise_mean = 0
noise_std = 0.5
noise_factor = 0.3

x_train = np.array(train_x)
x_test = np.array(test_x)
x_val = np.array(val_x)
train_x_noise = train_x + (noise_factor * np.random.normal(loc=noise_mean, scale=noise_std, size=x_train.shape) )
val_x_noise = val_x + (noise_factor * np.random.normal(loc=noise_mean, scale=noise_std, size=x_val.shape) )
test_x_noise = test_x + (noise_factor * np.random.normal(loc=noise_mean, scale=noise_std, size=x_test.shape) )

plt.figure(figsize=(8, 4))
for i in range(5):

  plt.subplot(2, 5, i+1)
  plt.imshow(x_test[i])
  plt.axis('off')

  plt.subplot(2, 5, (i+5)+1)
  plt.imshow(test_x_noise[i])
  plt.axis('off')

plt.show()
plt.savefig('before_predict.png')

input_img = Input(shape=(80,80,3))
x1 = Conv2D(256,(3,3),activation='relu',padding='same')(input_img)
x2 = Conv2D(128,(3,3),activation='relu',padding='same')(x1)
x3 = MaxPool2D(pool_size=(2,2),strides=2)(x2)
x4 = Conv2D(64,(3,3),activation='relu',padding='same')(x3)
x5 = Conv2D(64,(3,3),activation='relu',padding='same')(x4)
x6 = UpSampling2D(interpolation='nearest',size=(2,2))(x5)
x7 = Conv2D(128,(3,3),activation='relu',padding='same')(x6)
x8 = Conv2D(256,(3,3),activation='relu',padding='same')(x7)
decoded_img = Conv2D(3,(3,3),activation='relu',padding='same')(x8)
autoencoder = Model(input_img, decoded_img)
autoencoder.compile(optimizer=tf.keras.optimizers.Adadelta(learning_rate=0.01), loss='mse')
autoencoder.summary()

epochs = 2
batch_size = 16
callback = EarlyStopping(monitor='loss', patience=3)
history = autoencoder.fit(train_x_noise, x_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(val_x_noise, x_val), callbacks=[callback], verbose=1)
plt.figure(figsize=(10,10))
plt.plot(history.history['loss'],'b')
plt.plot(history.history['val_loss'],'r--',lw=2)
plt.title('Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.show()
plt.savefig('plot.png')
x_test_predicted = autoencoder.predict(test_x_noise)
plt.figure(figsize=(8,8))
n = 5
for i in range(n):
	plt.subplot(3, 5, i+1)
	plt.imshow(x_test[i])
	plt.title('Ground truth')
	plt.axis('off')

	plt.subplot(3, 5,n+ i+1)
	plt.imshow(test_x_noise[i])
	plt.title('Noisy image')
	plt.axis('off')

	plt.subplot(3, 5,2*n + i+1)
	plt.imshow(x_test_predicted[i])
	plt.title('Predicted image')
	plt.axis('off')
plt.tight_layout()
plt.show()
plt.savefig('predicted.png')