from cv2 import cv2
import numpy as np


# ---- IMAGE COMPRARISON ----
# percentage of similarity of two given images
img1 = cv2.imread('kot1.jpg')
img2 = cv2.imread('kot1a.jpg')

if img1.shape == img2.shape:
    difference = cv2.subtract(img1, img2)
    # if the whole image is black -> there is no difference between images
    r, g, b = cv2.split(difference)
    if cv2.countNonZero(r) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(b) == 0:
       print("images are equal")
    else:
        print("images are not equal")
        # percentage difference
        percentage = (np.count_nonzero(difference)*100) / difference.size
        print("they are different in: ", percentage, "%")
else:
    print("images are not the same size")


# ---- IMAGE INPAINTING ----
# repair damaged image by adding the mask and inpaint
img = cv2.imread('1.jpg')
mask = cv2.imread('1-mask.jpg', 0)

repaired = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)

# save output
cv2.imwrite('repaired1.jpg', repaired)



cv2.waitKey(0)
cv2.destroyAllWindows()
