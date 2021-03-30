# ReDust
This neural network has been designed as a part of a my bachelor's thesis. The goal was to create a model which would detect destroyed parts of the image. This, in turn, was to make it possible to apply the image inpainting process (cv2.inpaint()) in order to repair the missing information from pixels. Several types of neural network’s architecture have been checked. 

I started with the simple implementation of CNN architecture as it is highly useful in tasks connected with images and detecting particular objects on them. The main goal was to train the network to recognize damaged parts on the image so that it could generate black and white masks. There were multiply settings checked to find the best approach to the problem such as different values of batch size and epochs,  application of regularization, trying to overfit the network as well as making sure if training models on GPU and CPU makes any difference.  

The results have shown quite a good performance on the training dataset. Although it turned out, there was no big difference between models trained on 45 or 500 epochs. Another disturbing thing was that values of loss function were really high and values of accuracy very low.  These findings led me to investigate the mistakes in the code as well as improving the architecture. 

Thanks to the exploration and consultation, I introduced changes to the architecture as well as rescaled masks from training dataset. I decided to use UNet architecture as it is highly known for its detailed recognition of particular features. All of these changes resulted in accuracy values close to 1 for test images which was a great improvement to earlier versions.  

In further research, it may be promising to use GAN architecture instead of CNN. However, this would result in a change of the entire process of image reparation. The network would learn how to restore missing information from an image without generating masks. This means it would try to recognize what is on an image and try to imitate what could have been in a destroyed part. Another possibility is to apply different noise to the images and enlarge the size of images as I was training models on images with the size of 100x100 pix. This would lead to make the program more efficient and useful. 

# Settings
To run the code, create two folder in the same folder where the code is 'results' and 'models'. Folder with dataset should also be in the same folder as code.
