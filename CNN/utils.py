import os
import numpy as np
from cv2 import cv2
from PIL import Image
import matplotlib.pyplot as plt

from tensorflow import keras
from keras.preprocessing.image import array_to_img, img_to_array, load_img

PATH = os.getcwd()

## ----- LOAD DATA ------
def load_data(folderName, photoType):
    folderPath = os.path.join(PATH, 'data/', folderName, '')
    data = os.listdir(folderPath)
    dataList = []
    for sample in data:
        imgPath = folderPath + sample
        img = cv2.imread(imgPath, photoType)
        dataList.append(img)
    return np.array(dataList) / 255.


## ----- IMAGE AUGMENTATION -----
def dataset_augmentation(images_array):
    dataset = []
    for image in images_array:
            horizontal_flipped_img = cv2.flip(image, 1)
            dataset.append(horizontal_flipped_img)

            vertical_flipped_img = cv2.flip(image, 0)
            dataset.append(vertical_flipped_img)

            angles = [10, 15, 20]
            for angle in angles:
                height, width = image.shape[:2]
                matrix = cv2.getRotationMatrix2D((int(width/2), int(height/2)), angle, 1)
                rotated_img = cv2.warpAffine(image, matrix, (width, height))
                dataset.append(rotated_img)
    return np.array(dataset)


def plot_acc_loss(accuracy_values, loss_values, saving_name):
    # plot training ACCURACY VALUES
    fig = plt.figure()
    gs = fig.add_gridspec(ncols = 1,nrows = 2)
    plt.subplot(gs[0])
    plt.plot(accuracy_values)
    plt.ylabel('Accuracy')
    plt.legend(['Train data'], loc = 'upper left')

    # plot training LOSS VALUES
    plt.subplot(gs[1])
    plt.plot(loss_values)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train data'], loc = 'upper left')
    plt.tight_layout()
    fig.savefig(os.path.join(PATH, 'results','') + '{}.png'.format(saving_name), dpi = fig.dpi)


def plot_predictions(predictions, test_masks, saving_name):
    # plot 6 predicted masks and 6 original masks
    fig2 = plt.figure(constrained_layout = True)
    gs1 = fig2.add_gridspec(ncols = 3, nrows = 4)
    img_number = 0
    for row in range(4):
        for col in range(3):
            if row < 2:
                    mask_predicted = keras.preprocessing.image.array_to_img(predictions[img_number], scale = True)
                    fig2.add_subplot(gs1[row, col])
                    plt.axis('off')
                    plt.imshow(mask_predicted, cmap = 'gray')
            else:
                mask_original = keras.preprocessing.image.array_to_img(test_masks[img_number - 6].reshape((100, 100, 1)), scale = True)
                plt.subplot(gs1[row,col])
                plt.axis('off')
                plt.imshow(mask_original, cmap = 'gray')
            img_number += 1
    fig2.suptitle('Predicted masks (on top) vs original ones', fontsize = 16)
    fig2.savefig(os.path.join(PATH, 'results', '') + '{}_masks.png'.format(saving_name), dpi = fig2.dpi)

