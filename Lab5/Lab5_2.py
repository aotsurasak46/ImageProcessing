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

erum_image = cv2.imread("erum.png")
#กำหนดพารามิเตอร์ต่าง ๆ สำหรับ data augmentation
fill_method = ['constant','nearest','reflect','wrap']
npic = 4
rotation_range = 15
width_shift_range=0.1
height_shift_range=0.1
shear_range=0.2
zoom_range=0.2
horizontal_flip=True
mean_noise = 0.2
std = 5
scale_factor = 5

#สร้างฟังก์ชั่นเพื่อเพิ่ม noise ให้กับรูปภาพ
def add_gaussian_noise(img):
  noise = np.random.normal(loc=mean_noise, scale=std,size=img.shape)
  img_noisy = img+ noise * scale_factor
  return img_noisy

#สร้าง VideoWrter สำหรับเขียนวิดีโอ
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
frame_rate = 1.0  # 1 frame per second
out = cv2.VideoWriter('output_video.mp4', fourcc, frame_rate, (erum_image.shape[1], erum_image.shape[0]))
all_result_array = []
#loop สำหรับแต่ละ fill_method
for m in range(len(fill_method)):
  # สร้าง ImageDataGenerator ด้วยพารามิเตอร์ที่กำหนดไว้
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
  #เตรียมรูปภาพ input 
  img_batch = np.expand_dims(erum_image,axis=0)
  pic = datagen.flow(img_batch,batch_size=1)
  for i in range(1, npic+1):
        batch = pic.next()
        im_result = batch[0].astype('uint8')
        plt.subplot(m+1, 4, (i)+(m*4))
        batch = pic.next()
        im_result = batch[0].astype('uint8')
        all_result_array.append(im_result)
        plt.imshow(cv2.cvtColor(im_result, cv2.COLOR_BGR2RGB))
  plt.show()

#นำแต่ละรูปมาทำวิดีโอ
for i in range(len(all_result_array)):
  out.write(all_result_array[i])

out.release()
