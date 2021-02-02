import keras
import tensorflow as tf
from keras.layers import Conv2D,Activation,MaxPooling2D,Dense,Flatten,Dropout,AveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from PIL import Image
from keras.callbacks import ModelCheckpoint, EarlyStopping
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from keras.layers.normalization import BatchNormalization
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os
from keras.models import load_model


warnings.filterwarnings(
    action='ignore',
    category=UserWarning,
    module=r'.*TiffImagePlugin'
)

size = 224
test_data_dir = "TEST"
batch_size = 32

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(size, size),
    batch_size=batch_size,
    class_mode='categorical', 
    shuffle=False) # It must be FALSE in test_generator


def loadmodel(filepath):
    filename = os.path.join(filepath)
    try:
        model = tf.keras.models.load_model(filename)
        print("\nModel loaded successfully from file %s\n" %filename)
    except OSError:    
        print("\nModel file %s not found!!!\n" %filename)
        model = None
    return model

model_n = loadmodel('1601997.h5')

val_steps=test_generator.n//test_generator.batch_size+1
loss, acc = model_n.evaluate_generator(test_generator,verbose=1,steps=val_steps)
print('Test loss: %f' %loss)
print('Test accuracy: %f' %acc)


preds = model_n.predict_generator(test_generator,verbose=1,steps=val_steps)
Ypred = np.argmax(preds, axis=1)
Ytest = test_generator.classes
print(classification_report(Ytest, Ypred, labels=None, target_names=["Haze","Rainy","Snowy","Sunny"], digits=3))