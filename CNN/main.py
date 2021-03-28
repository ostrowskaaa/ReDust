import os
import numpy as np

import tensorflow as tf
from tensorflow import keras
import keras.backend as K

from model import get_model
from utils import load_data, dataset_augmentation, plot_acc_loss, plot_predictions

PATH = os.getcwd()
BATCH_SIZE = [32]
EPOCHS_NUMBER = [500]
LOSS = 'categorical_crossentropy'
SAVING_NAME = 'model{}_e{}_b{}.h5'.format(LOSS, EPOCHS_NUMBER, BATCH_SIZE)

# ------  DATASETS  ------
# images are scaled in order to lower the numbers that CNN is processing
# 1 indicates color photo, 0 BW photo
train_images = load_data('train_images', 1) 
train_masks = load_data('train_masks', 0)
test_images = load_data('test_images', 1) 
test_masks = load_data('test_masks', 0) 

# Transforming masks to their binary versions
train_masks = np.ceil(train_masks)
test_masks = np.ceil(test_masks)

# connect original images with the augmented ones
train_images = np.concatenate((train_images, dataset_augmentation(train_images))) # 3744 images now
train_masks = np.concatenate((train_masks, dataset_augmentation(train_masks)))

# ------- MODEL TRAINING ----------
model = get_model()

model.compile(optimizer='adam',
                loss=[LOSS],
                metrics=['accuracy'])

fit = model.fit(train_images, train_masks, batch_size = BATCH_SIZE, epochs = EPOCHS_NUMBER)
test_loss, test_acc = model.evaluate(test_images, test_masks, verbose = 2)
summary = model.summary()

# output into array with shape (100,100,1)
predictions = model.predict(test_images).reshape((len(test_images), 100, 100, 1))

# save model, loss values, accuracy values and predicted masks on computer
model_path = os.path.join(PATH, 'models', '')
model.save(model_path + SAVING_NAME)

results_file = open('models/{}test_loss{}test_accuracy.txt'.format(test_loss, test_acc), 'w')
results_file.write(SAVING_NAME)
results_file.close()

plot_acc_loss(fit.history['accuracy'], fit.history['loss'], SAVING_NAME)
plot_predictions(predictions, test_masks, SAVING_NAME)
