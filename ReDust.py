import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten


def LoadColorData(folderName):
    PATH = os.getcwd()
    folder_path = PATH + '/data/' + folderName + '/'
    data = os.listdir(folder_path)
    x_data = []
    for sample in data:
        img_path = folder_path + sample
        x = cv2.imread(img_path,1)
        # rescale so there are only values between 0 and 1
        x = x / 255.0
        x_data.append(x)

    x_data = np.array(x_data)
    return x_data

def LoadBWData(folderName):
    PATH = os.getcwd()
    folder_path = PATH + '/data/' + folderName + '/'
    data = os.listdir(folder_path)
    x_data = []
    for sample in data:
        img_path = folder_path + sample
        x = cv2.imread(img_path, 0)
        x_data.append(x)

    x_data = np.array(x_data)
    return x_data

# load training data
train = 'train'
train_images = LoadColorData(train)

# load training masks
train_mask = 'train_mask'
train_mask = LoadBWData(train_mask)


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(100,100,3)),
    keras.layers.Dense(32),
    keras.layers.Dropout(0.5),
    keras.layers.Activation('sigmoid'),
    keras.layers.Dense(10)
])

summary = model.summary()
print (summary)

model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

model.fit(train_images, train_mask, batch_size=32, epochs=1)

eval_model=classifier.evaluate(train_images, train_mask, verbose=2)

print('Test accuracy:', eval_model)
