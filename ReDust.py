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
    keras.layers.Flatten(input_shape=(100,100,3)),
    keras.layers.Dense(32),
    keras.layers.Dropout(0.5),
    keras.layers.Activation('sigmoid'),
    keras.layers.Dense(10000),
    keras.layers.Reshape((100,100))
])

summary = model.summary()
print (summary)

model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

fit = model.fit(train_images, train_mask, batch_size=32, epochs=1)

eval_model = model.evaluate(train_images, train_mask, verbose=2)

print('Test accuracy:', eval_model)

#probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = model.predict(train_images)
data = []
for i in predictions:
    img = i.reshape((100, 100, 1))
    data.append(img)

img = keras.preprocessing.image.array_to_img(data[0], scale=True)
print(type(img))
from PIL import Image
img.save('trial.png')
im = Image.open('trial.png')
im.show()



'''
def plot_image(i, predictions_array, img):
  predictions_array, img = predictions_array[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, predictions_array)

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], train_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
plt.tight_layout()
plt.show()
'''
