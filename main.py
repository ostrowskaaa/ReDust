import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten
from keras.preprocessing.image import img_to_array, array_to_img


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
    keras.layers.Dense(30000),
    keras.layers.Conv2D(16, (3,3), padding='same', input_shape=(100,100, 3)),
    keras.layers.Activation('relu'),
    keras.layers.Dropout(0.4),
    keras.layers.Conv2D(16, (3,3), padding='same', input_shape=(100,100, 3)),
    keras.layers.Activation('relu'),
    keras.layers.Dropout(0.4),
    keras.layers.Flatten(input_shape=(100,100,3)),
    keras.layers.Dense(10000, activation='sigmoid'),
    keras.layers.Reshape((100,100))
])


model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])


fit = model.fit(train_images, train_mask, batch_size=32, epochs=1)


eval_model = model.evaluate(train_images, train_mask, verbose=2)


summary = model.summary()


predictions = model.predict(train_images)
data = []
for i in predictions:
    img = i.reshape((100, 100, 1))
    data.append(img)

img = keras.preprocessing.image.array_to_img(data[0], scale=True)
#print(type(img))

text_file = open('outputData.txt', 'w')
text_file.write(summary)
text_file.write('\n\n\n')
text_file.write('accuracy: ', eval_model)
text_file.close()


from PIL import Image
img.save('trial.png')
im = Image.open('trial.png')
im.show()
