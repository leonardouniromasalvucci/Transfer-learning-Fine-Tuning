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
import numpy as np
import os
from keras.models import load_model

warnings.filterwarnings(
    action='ignore',
    category=UserWarning,
    module=r'.*TiffImagePlugin'
)

size = 224
test_data_dir = "BLIND_TEST"
batch_size = 32

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(size, size),
    class_mode=None,
    subset=None,
    classes=None,
    batch_size=batch_size, 
    shuffle=False)


def loadmodel(problem):
    filename = os.path.join(problem)
    try:
        model = tf.keras.models.load_model(filename)
        print("\nModel loaded successfully from file %s\n" %filename)
    except OSError:    
        print("\nModel file %s not found!!!\n" %filename)
        model = None
    return model

model_n = loadmodel('1601997.h5')

from sklearn.metrics import classification_report, confusion_matrix

preds = model_n.predict(test_generator)
Ypred = np.argmax(preds, axis=1)

with open('1601997.csv', 'w') as csv_file:
    for i in range(len(Ypred)):
        if (Ypred[i]==0):
            line = "HAZE"+"\n"
        elif (Ypred[i]==1):
            line = "RAINY"+"\n"
        elif (Ypred[i]==2):
            line = "SNOWY"+"\n"
        elif (Ypred[i]==3):
            line = "SUNNY"+"\n"
        else:
            line = "\n"

        csv_file.write(line)

