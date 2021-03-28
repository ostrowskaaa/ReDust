import os
import numpy as np
from cv2 import cv2
from PIL import Image
import matplotlib.pyplot as plt

from tensorflow import keras
from keras.preprocessing.image import array_to_img, img_to_array, load_img

PATH = os.getcwd()

## ----- LOAD DATA ------
def load_data(folder_name, photo_type = 1):
    PATH = os.getcwd()
    # folder_path = PATH + '/data/' + folderName + '/'
    folder_path = f'{PATH}/data/{folder_name}/'
    data = os.listdir(folder_path)
    x_data = []
    for sample in data:
        img_path = folder_path + sample
        x = cv2.imread(img_path, 1)
        x = x / 255.0
        x_data.append(x)
    return np.array(x_data)

## ----- IMAGE TRANSFORMATION -----
def dataset_transformation(images_array):
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
