import cv2
import numpy as np

from matplotlib import pyplot as plt

# read image
img = cv2.imread('./image/original.png')

# convert the image into grayscale before doing histogram equalization
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# image equalization
equalize_img = cv2.equalizeHist(gray_img)

# create clahe image
clahe = cv2.createCLAHE()
clahe_img = clahe.apply(gray_img)

# show image
cv2.imshow("image", gray_img)
cv2.imshow("equal_image", equalize_img)
cv2.imshow("clahe_image", clahe_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# plot image histogram 
plt.hist(gray_img.ravel(), 256, [0, 255],label= 'original image')
plt.hist(equalize_img.ravel(), 256, [0, 255],label= 'equalize image')
plt.hist(clahe_img.ravel(), 256, [0, 255],label= 'clahe image')
plt.legend()