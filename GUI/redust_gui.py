# .ckpt files with model should be in the same folder as redust_gui.py
# 'bce_e1500_b32model.ckpt' is a model from UNet network trained on 500 epochs
#  with batch size 32, it was one of the best models from the training process
# mask and repaired image will be saved in the same folder as redust_gui.py

import os
from cv2 import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import array_to_img, img_to_array, load_img, save_img
from keras.layers import *
from keras.models import *
import keras.backend as K
from models import UNet
from metrics import F1Score


PATH = os.getcwd()


class App(tk.Frame):
    def __init__(self, master = None):
        super().__init__(master)
        self.master = master
        self.arrayImage = []
        self.destroyedImg = None
        self.mask = None
        self.repaired = None

    def load_img(self):
        if len(self.arrayImage) != 0:
            self.arrayImage.pop(0)
        path = filedialog.askopenfilename(filetypes = (("Image Files", "*.JPG"), ("All files", "*")))
        img = cv2.imread(path, 1)
        self.destroyedImg = img
        self.arrayImage.append(img / 255.)
        App.display(self)

    def predict_mask(self):
        prediction = model.predict(np.array(self.arrayImage))
        prediction = np.array(prediction)
        img = save_img(PATH + '/mask.jpg', prediction.reshape(100, 100, 1))
        self.mask = cv2.imread(PATH + '/mask.jpg', 0)
        App.display(self)

    def inpainting(self):
        img = self.destroyedImg
        mask = self.mask
        self.repaired = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)
        cv2.imwrite('repaired.jpg', self.repaired)
        App.display(self)

    def display(self):
        if self.destroyedImg is not None:
            b, g, r = cv2.split(self.destroyedImg)
            img = Image.fromarray(cv2.merge((r, g, b)))
            tkImage = ImageTk.PhotoImage(image = img)
            canvas1.configure(image = tkImage)
            canvas1.image = tkImage
        if self.mask is not None:
            tkImage = ImageTk.PhotoImage(image = Image.fromarray(self.mask))
            canvas2.configure(image = tkImage)
            canvas2.image = tkImage
        if self.repaired is not None:
            b, g, r = cv2.split(self.repaired)
            img = Image.fromarray(cv2.merge((r, g, b)))
            tkImage = ImageTk.PhotoImage(image = img)
            canvas3.configure(image = tkImage)
            canvas3.image = tkImage

            
################ CHOOSE UNET MODEL FOR PREDICTIONS
model = UNet()
model.compile(optimizer = 'adam',
            loss = ['bce'],
            metrics = ['bce', 'accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), F1Score()])
model.load_weights('bce_e1500_b32model.ckpt')
##################################################

############### INITIATE THE GUI
window = tk.Tk()
window.title('ReDust')
window.geometry('450x300')

canvas1 = tk.Label(window)
canvas1.place(x = 55, y = 100)
canvas2 = tk.Label(window)
canvas2.place(x = 175, y = 100)
canvas3 = tk.Label(window)
canvas3.place(x = 295, y = 100)

app = App(master = window)
############## DEFINE BUTTONS
load = tk.Button(window, text = 'load image', command = lambda: app.load_img())
load.place(x = 75, y = 50)
predict = tk.Button(window, text = 'predict the mask', command = lambda: app.predict_mask())
predict.place(x = 180, y = 50)
inpaint = tk.Button(window, text = 'inpaint', command = lambda: app.inpainting())
inpaint.place(x = 325, y = 50)
##############################################
app.mainloop()

cv2.waitKey(0)
cv2.destroyAllWindows()
