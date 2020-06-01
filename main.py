import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from operator import itemgetter, attrgetter

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
    return np.array(x_data)

def LoadBWData(folderName):
    PATH = os.getcwd()
    folder_path = PATH + '/data/' + folderName + '/'
    data = os.listdir(folder_path)
    x_data = []
    for sample in data:
        img_path = folder_path + sample
        x = cv2.imread(img_path, 0)
        x_data.append(x)
    return np.array(x_data)

####################################################################

# load  images
data_images = 'data_images'
data_images = LoadColorData(data_images)
# divide data into train and test
train_images, test_images = np.split(data_images, [int(len(data_images)*0.8)])
print('\ntreningowe: ', len(train_images), '\ntestowe: ', len(test_images))

# load  masks
data_mask = 'data_mask'
data_mask = LoadBWData(data_mask)
# divide data into train and test
train_mask, test_mask = np.split(data_mask, [int(len(data_mask)*0.8)])

#######################

# training loop for different options
loss_options = ['categorical_crossentropy', 'binary_crossentropy']
batch_options = [16, 32, 64, 128]
epochs_options = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 95, 100]


for epochs in epochs_options:
    for batch in batch_options:
        for loss in loss_options:
            epochs_string = str(epochs)
            batch_string = str(batch)


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
                            loss=loss,
                            metrics=['accuracy'])


            fit = model.fit(train_images, train_mask, validation_split=0.2, batch_size=batch, epochs=epochs)
            test_loss, test_acc = model.evaluate(test_images, test_mask, verbose=2)
            summary = model.summary()

            # output into array with shape (100,100,1)
            predictions = model.predict(test_images)
            prediction_array = []
            for i in predictions:
                img_array = i.reshape((100, 100, 1))
                prediction_array.append(img_array)

            # save model on computer
            name = 'models/'+'epochs'+epochs_string+'batch'+batch_string+loss+'.h5'
            model.save(name)
            file = open('models/'+str(test_loss)+".txt", "w")
            file.write(name)
            file.close()


            # Plot training & validation ACCURACY VALUES
            fig = plt.figure()
            plt.subplot(121)
            plt.plot(fit.history['accuracy'])
            plt.plot(fit.history['val_accuracy'])
            plt.title('Model accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper left')

            # plot training & validation LOSS VALUES
            plt.subplot(122)
            plt.plot(fit.history['loss'])
            plt.plot(fit.history['val_loss'])
            plt.title('Model loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper left')
            fig.savefig('results/'+'epochs'+epochs_string+'batch'+batch_string+loss+'.png', dpi=fig.dpi)


            # plot 6 predicted masks
            fig2 = plt.figure(figsize=(2,3))
            rows = 2
            for j in range(6):
                img = keras.preprocessing.image.array_to_img(prediction_array[j], scale=True)
                plt.subplot(rows,3,j+1)
                plt.axis('off')
                plt.imshow(img, cmap='binary')
            fig2.savefig('results/'+'epochs'+epochs_string+'batch'+batch_string+loss+'_mask.png', dpi=fig.dpi)
