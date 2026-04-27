import os
from tempfile import template
import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

base_folder_path = os.path.dirname(os.path.abspath(__file__))
img_folder_path = os.path.join(base_folder_path, 'testPic')

img1 = cv2.imread(img_folder_path + '/111.png', cv2.IMREAD_GRAYSCALE)
# plt.imshow(template), plt.show()
img2 = cv2.imread(img_folder_path + '/bothClose.png', cv2.IMREAD_GRAYSCALE)
# plt.imshow(img), plt.show()


sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append(m)

img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, flags = cv2.DrawMatchesFlags_DEFAULT)

cv2.imshow('SIFT', img3)

cv2.waitKey(0)
cv2.destroyAllWindows()