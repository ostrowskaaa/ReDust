import os
import numpy as np
from cv2 import cv2

import tensorflow as tf
from tensorflow import keras
import keras.backend as K

from models import UNet
from metrics import F1Score
from utils import load_data, dataset_augmentation, plot_acc_loss

# Comment this part if it causes any errors.
#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Define training hyperparameters
PATH = os.getcwd()
BATCH_SIZE = 128
EPOCHS_NUMBER = 1
LOSS = 'bce'
SAVING_NAME = 'model{}_e{}_b{}'.format(LOSS, EPOCHS_NUMBER, BATCH_SIZE)

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

model = UNet()
model.compile(optimizer='adam',
            loss=[LOSS],
            metrics=['bce', 'accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), F1Score()])

# Train the model
fit = model.fit(train_images, train_masks, batch_size=BATCH_SIZE, epochs=EPOCHS_NUMBER, shuffle=True, validation_data=(test_images, test_masks))
test_loss, _, test_acc, _, _, _  = model.evaluate(test_images, test_masks)

# save model
model.save_weights('models/{}model.ckpt'.format(SAVING_NAME))

# Predict and save masks for the test set
predicted_masks = model.predict(test_images, batch_size=BATCH_SIZE)

for i in range(10):
    pred_mask = np.squeeze(predicted_masks[i])
    true_mask = test_masks[i]
    cv2.imwrite('results/{}{}.{}.png'.format(EPOCHS_NUMBER, BATCH_SIZE, i), np.ceil(np.concatenate((true_mask, pred_mask), axis=1) * 255.))

results_file = open('models/{}test_loss{}test_accuracy.txt'.format(test_loss, test_acc), 'w')
results_file.write(SAVING_NAME)
results_file.close()

plot_acc_loss(fit.history['accuracy'], fit.history['loss'], SAVING_NAME)

