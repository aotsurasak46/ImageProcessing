import numpy as np
import cv2
from matplotlib import pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D

base_model=MobileNet(weights='imagenet',include_top=False, input_shape=(224, 224, 3))
x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x) # dense layer 1
x=Dense(512,activation='relu')(x) #dense layer 2
x=Dense(512,activation='relu')(x) #dense layer 3
preds=Dense(3,activation='softmax')(x)
model=Model(inputs=base_model.input,outputs=preds)
EP = 300  #Epochs
BS = 32   #Batch Size
# model.summary()

for layer in model.layers[:86]:
  layer.trainable=False #Freeze base model
for layer in model.layers[86:]:
  layer.trainable=True #Unfreeze new added denses
  
# model.summary()
from tensorflow.keras.applications.mobilenet import preprocess_input
rotation_range = 90
width_shift_range=0.4
height_shift_range=0.7
shear_range=0.5
zoom_range=0.3
horizontal_flip=True
datagen=ImageDataGenerator( rotation_range=rotation_range, 
                            zoom_range=zoom_range,
                            width_shift_range=width_shift_range, 
                            height_shift_range=height_shift_range,
                            shear_range=shear_range, 
                            horizontal_flip=horizontal_flip,
                            preprocessing_function=preprocess_input,
                            fill_mode="nearest")


train_generator=datagen.flow_from_directory('./Train/', # this is where you specify the path to the main data folder
                            target_size=(224,224), color_mode='rgb',
                            batch_size=BS,
                            class_mode='categorical', 
                            seed= 42,
                            shuffle=True)
val_generator=datagen.flow_from_directory('./Validate/', # this is where you specify the path to the main data folder
                            target_size=(224,224), color_mode='rgb',
                            batch_size=BS,
                            class_mode='categorical', 
                            seed= 42,
                            shuffle=True)

batch1 = train_generator.next()
Img_train = batch1[0]
Img_train = (Img_train + 1.0) / 2.0

batch2 = val_generator.next()
Img_val = batch2[0]
Img_val = (Img_val + 1.0) / 2.0


opts = Adam(learning_rate = 0.0001)
model.compile(
              optimizer=opts,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

step_size_train=train_generator.n//train_generator.batch_size
step_size_val = val_generator.n//val_generator.batch_size

history=model.fit_generator(generator=train_generator,
                            steps_per_epoch=step_size_train,
                            validation_data = val_generator,
                            validation_steps = step_size_val,
                            epochs=EP,
                            verbose = 1)
N = list(range(1, EP + 1))
plt.plot(N, history.history["accuracy"], label="Train_acc")
plt.plot(N, history.history["val_accuracy"], label="Validate_acc")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('acc_lab6_2.png')  # Save the accuracy plot as an image
plt.close()  # Close the current figure to clear the plot

plt.plot(N, history.history['loss'], label="Train_loss")
plt.plot(N, history.history['val_loss'], label="Validate_loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_lab6_2.png')  

test_generator = datagen.flow_from_directory(
                                          "./Test/",
                                          class_mode="categorical",
                                          target_size=(224, 224), color_mode="rgb",
                                          shuffle=False,
                                          batch_size=1)
y_true = test_generator.classes
preds = model.predict_generator(test_generator)
print(preds.shape)
print(preds)
y_pred = np.argmax(preds,axis=1)
print(test_generator.classes)
print(y_pred)
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))