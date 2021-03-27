import os
import numpy as np
from cv2 import cv2
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.layers import *
from keras.models import *
import keras.backend as K
from models import UNet
from metrics import F1Score

# Comment this part if it causes any errors.
#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

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
            M = cv2.getRotationMatrix2D((int(w / 2), int(h / 2)), j, 1)
            img3 = cv2.warpAffine(i, M, (w, h))
            dataset.append(img3)
    return np.array(dataset)


# Loading the data
train_images = load_data('train_images')
train_masks = load_bw_data('train_masks') / 255.
test_images = load_data('test_images')
test_masks = load_bw_data('test_masks') / 255.

# Transforming masks to their binary versions
train_masks = np.ceil(train_masks)
test_masks = np.ceil(test_masks)

# connect original images with the augmented ones
# train_images = np.concatenate((train_images, dataset_augmentation(train_images)))  # 3744 images now
# train_masks = np.concatenate((train_masks, dataset_augmentation(train_masks)))
# test_images = np.concatenate((test_images, dataset_augmentation(test_images)))
# test_masks = np.concatenate((test_masks, dataset_augmentation(test_masks)))

# Define training hyperparameters
batch_size = [32, 64]
epochs = [300, 400, 600]
loss = 'bce'

model = UNet()

for epochs in epochs:
    for batch_size in batch_size:
        
        saving_name = loss + '_e' + str(epochs) + '_b' + str(batch_size)

        model.compile(optimizer='adam',
                    loss=[loss],
                    metrics=['bce', 'accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), F1Score()])

        # Train the model
        fit = model.fit(train_images, train_masks, batch_size=batch_size, epochs=epochs, shuffle=True, validation_data=(test_images, test_masks))
        test_loss, _, test_acc, _, _, _  = model.evaluate(test_images, test_masks)
        
        # save model
        model.save_weights('models/' + saving_name + 'model.ckpt')

        # Predict masks for the test set
        predicted_masks = model.predict(test_images, batch_size=32)

        # Uncomment to save your predictions to the folder specified below
        for i in range(10):
            pred_mask = np.squeeze(predicted_masks[i])
            true_mask = test_masks[i]
            cv2.imwrite("results/" + str(epochs) + str(batch_size) + ".{}.png".format(i), np.ceil(np.concatenate((true_mask, pred_mask), axis=1) * 255.))


        file = open('models/'+str(test_loss)+'test_loss'+str(test_acc)+'test_accuracy'+".txt", "w")
        file.write(saving_name)
        file.close()


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
