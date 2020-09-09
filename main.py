import os
import numpy as np
from cv2 import cv2
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten
import keras.backend as K

batch_size = 64
epochs = 1
loss='categorical_crossentropy'
saving_name = loss + '_e' + str(epochs) + '_b' + str(batch_size)

## ----- LOAD DATA ------
def load_data(folderName):
    PATH = os.getcwd()
    folder_path = PATH + '/data/' + folderName + '/'
    data = os.listdir(folder_path)
    x_data = []
    for sample in data:
        img_path = folder_path + sample
        x = cv2.imread(img_path, 1)
        x = x / 255.0
        x_data.append(x)
    return np.array(x_data)

def load_bw_data(folderName):
    PATH = os.getcwd()
    folder_path = PATH + '/data/' + folderName + '/'
    data = os.listdir(folder_path)
    x_data = []
    for sample in data:
        img_path = folder_path + sample
        x = cv2.imread(img_path, 0)
        x_data.append(x)
    return np.array(x_data)

## ----- IMAGE TRANSFORMATION -----
def dataset_augmentation(array):
# horizontal_flip
        dataset = []
        for i in array:
                img1 = cv2.flip(i, 1)
                dataset.append(img1)
# vertical_flip
                img2 = cv2.flip(i, 0)
                dataset.append(img2)
# rotation
                angle = [10, 15, 20]
                for j in angle:
                    h, w = i.shape[:2]
                    M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), j, 1)
                    img3 = cv2.warpAffine(i, M, (w, h))
                    dataset.append(img3)
        return np.array(dataset)

## ----- ACCURACY -----
def tensorToNumpy(y_true, y_pred):
    return tf.numpy_function(custom_acc, (y_true, y_pred), tf.double)

def custom_acc(y_true, y_pred):
    results_array = []
    for i in range(y_true.shape[0]):
        #print('zdjcie: ', i)
        count_white = 0
        count_equal_white = 0
        for x in np.nditer([y_true[i], y_pred[i]]):
            if x[0] != 0:
                count_white += 1
                if x[1] != 0:
                    count_equal_white += 1
        #print(count_white)
        #print(count_equal_white)
        results_array.append((count_equal_white / count_white) *100)
    return np.array(results_array)

def categorical_accuracy(y_true, y_pred):
    return K.cast(K.equal(K.argmax(y_true, axis=-1),
                          K.argmax(y_pred, axis=-1)),
                          K.floatx()) 

##################      ------  PROGRAM  ------         ###################
train_images = load_data('train_images')
train_masks = load_bw_data('train_masks')
test_images = load_data('test_images')
test_masks = load_bw_data('test_masks')

# connect original images with the augmented ones
train_images = np.concatenate((train_images, dataset_augmentation(train_images))) # ~3744 images now
train_masks = np.concatenate((train_masks, dataset_augmentation(train_masks)))
test_images = np.concatenate((test_images, dataset_augmentation(test_images)))
test_masks = np.concatenate((test_masks, dataset_augmentation(test_masks)))

## ----- MODEL -----
model = keras.Sequential([
                keras.layers.Conv2D(32, (3,3), input_shape=(100,100, 3), activation='relu'),
                keras.layers.MaxPooling2D(pool_size=(2, 2)),
                keras.layers.Conv2D(64, (3,3), activation='relu'),
                keras.layers.MaxPooling2D(pool_size=(2, 2)),
                keras.layers.Conv2D(128, (3,3), activation='relu'),
                keras.layers.MaxPooling2D(pool_size=(2, 2)),
                keras.layers.Flatten(),
                keras.layers.Dense(10000, activation='sigmoid'),
                keras.layers.Reshape((100,100))
            ])

model.compile(optimizer='adam',
                loss=[loss],
                metrics=['accuracy'])

fit = model.fit(train_images, train_masks, batch_size=batch_size, epochs=epochs)
test_loss, test_acc = model.evaluate(test_images, test_masks, verbose=2)
summary = model.summary()

# output into array with shape (100,100,1)
predictions = model.predict(test_images)
prediction_array = []
for i in predictions:
    img_array = i.reshape((100, 100, 1))
    prediction_array.append(img_array)

# save model on computer
name = 'models/'+saving_name+'.h5'
model.save(name)
file = open('models/'+saving_name+".txt", "w")
file.write(name)
file.close()

# plot 6 predicted masks and 6 original masks
fig2 = plt.figure(constrained_layout=True)
gs1 = fig2.add_gridspec(ncols=3,nrows=4)
n = 0
for row in range(4):
    for col in range(3):
        if row < 2:
            mask_predicted = keras.preprocessing.image.array_to_img(prediction_array[col], scale=True)
            fig2.add_subplot(gs1[row,col])
            plt.axis('off')
            plt.imshow(mask_predicted, cmap='gray')
            n+=1
        else:
            mask_original = keras.preprocessing.image.array_to_img(test_masks[n-6].reshape((100,100,1)), scale=True)
            plt.subplot(gs1[row,col])
            plt.axis('off')
            plt.imshow(mask_original, cmap='gray')
            n+=1
fig2.suptitle('Predicted masks (on top) vs original ones', fontsize=16)
fig2.savefig('results/'+saving_name+'_masks.png', dpi=fig2.dpi)

# plot training ACCURACY VALUES
fig = plt.figure()
gs = fig.add_gridspec(ncols=1,nrows=2)
plt.subplot(gs[0])
plt.plot(fit.history['accuracy'])
plt.ylabel('Accuracy')
plt.legend(['Train data'], loc='upper left')

# plot training LOSS VALUES
plt.subplot(gs[1])
plt.plot(fit.history['loss'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train data'], loc='upper left')
plt.tight_layout()
fig.savefig('results/'+saving_name+'.png', dpi=fig.dpi)
