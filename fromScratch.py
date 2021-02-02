import tensorflow.keras as keras
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D,Activation,MaxPooling2D,Dense,Flatten,Dropout,AveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
from keras.callbacks import ModelCheckpoint, EarlyStopping
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
from keras import optimizers, regularizers
import os
from keras.models import load_model


warnings.filterwarnings(
    action='ignore',
    category=UserWarning,
    module=r'.*TiffImagePlugin'
)

size = 224
train_data_dir = "TRAIN"
val_data_dir = "VALIDATION"
test_data_dir = "TEST"
batch_size = 64
epochs = 100


train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    fill_mode="nearest",
    zoom_range=0.3,
    width_shift_range=0.3,
    height_shift_range=0.3,
    rotation_range=30)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    #interpolation = 'lanczos',
    target_size=(size, size),
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True)
    #save_to_dir='ac')

validation_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    fill_mode="nearest",
    zoom_range=0.3,
    width_shift_range=0.3,
    height_shift_range=0.3,
    rotation_range=30)

validation_generator = validation_datagen.flow_from_directory(
    val_data_dir,
    #interpolation = 'lanczos',
    target_size=(size, size),
    class_mode="categorical",
    batch_size=batch_size,
    shuffle=True)



def MyNet(input_shape, num_classes):

    model = Sequential()

    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=input_shape, padding="same"))    
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    
    model.add(Conv2D(64, kernel_size=(4, 4), strides=(1, 1), activation='relu', padding='valid'))    
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    
    model.add(Conv2D(128, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='valid'))
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    model.add(Conv2D(128, kernel_size=(4, 4), strides=(1, 1), activation='relu', padding='valid'))
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    return model


model = MyNet((size,size,3),4)
model.compile(loss="categorical_crossentropy",optimizer='adamax',metrics=['accuracy'])

steps_per_epoch=train_generator.n//train_generator.batch_size
val_steps=validation_generator.n//validation_generator.batch_size+1

checkpoint = ModelCheckpoint("NetFromScratch.h5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min', period=1)
early = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')


callbacks_list = [checkpoint, early]

history = model.fit_generator(
    train_generator,
    workers=6,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    validation_data=validation_generator,
    callbacks=callbacks_list)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()