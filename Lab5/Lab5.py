# Import Libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation, Flatten, Conv2D, MaxPooling2D
from keras import Model, Input
import keras.utils as image
from keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, UpSampling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers
from tensorflow.keras.datasets import fashion_mnist

from sklearn.model_selection import train_test_split

x_train,x_test = fashion_mnist.load_data()
x_train = x_train/255
x_test = x_test/255

x_train,x_test = train_test_split(x_train,random_state=25,test_size=0.3)
x_train,x_val = train_test_split(x_train,random_state=13,test_size=0.2)

rotation_range = 15
width_shift_range=0.1
height_shift_range=0.1
shear_range=0.2
zoom_range=0.2
horizontal_flip=True

mean_noise = [0.2,0.8]
std = [5,10]
scale_factor = [5,4]
fill_method = ['constant','nearest','reflect','wrap']

def add_gaussian_noise(img,mean_noise,std,scale_factor):
  noise = np.random.normal(loc=mean_noise[0], scale=std[0],size=img.shape)
  img_noisy = img+ noise * scale_factor[0]
  return img_noisy

for m in range(len(fill_method)):
  datagen = ImageDataGenerator(
                              rotation_range=rotation_range,
                               width_shift_range=width_shift_range,
                               height_shift_range=height_shift_range,
                               shear_range = shear_range,
                               zoom_range = zoom_range,
                               horizontal_flip=horizontal_flip,
                               preprocessing_function=add_gaussian_noise,
                               fill_mode = fill_method[m]
                               )

def create_model(optimizer='adam', learning_rate=0.01):
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
    opt = None  # Initialize opt to None

    if optimizer == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'adadelta':
        opt = tf.keras.optimizers.Adadelta(learning_rate=learning_rate)
    if opt is not None:
        autoencoder.compile(optimizer=opt, loss='mse')
    else:
        raise ValueError("Invalid optimizer choice")
    return autoencoder

# train with best parameter
opts = 'adam'
lr = 0.01
epoch = 70
batchsize = 16

autoencoder = create_model(optimizer=opts,learning_rate=lr)
callback = EarlyStopping(monitor='loss', patience=3)
history = autoencoder.fit_generator(datagen.flow(x_train,x_train,batch_size = batchsize), epochs=epoch,steps_per_epoch=x_train.shape[0]//batchsize, batch_size=batchsize, shuffle=True, validation_data=datagen.flow(x_val,x_val,batch_size=batchsize), callbacks=[callback], verbose=1)
x_test_noisy = add_gaussian_noise(x_test)
plt.figure(figsize=(10,10))
plt.plot(history.history['loss'],'b')
plt.plot(history.history['val_loss'],'r--',lw=2)
plt.title('Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.show()
plt.savefig('plot_lab5.png')
predict_test = autoencoder.predict(x_test_noisy)
plt.figure(figsize=(8,8))
n = 5
for i in range(n):
	plt.subplot(3, 5, i+1)
	plt.imshow(x_test[i])
	plt.title('Ground truth')
	plt.axis('off')

	plt.subplot(3, 5,n+ i+1)
	plt.imshow(x_test_noisy[i])
	plt.title('Noisy image')
	plt.axis('off')

	plt.subplot(3, 5,2*n + i+1)
	plt.imshow(predict_test[i])
	plt.title('Predicted image')
	plt.axis('off')
plt.tight_layout()
plt.show()
plt.savefig('predicted_lab5.png')