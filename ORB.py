import os
import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

base_folder_path = os.path.dirname(os.path.abspath(__file__))
img_folder_path = os.path.join(base_folder_path, 'testPic')

template = cv2.imread(img_folder_path + '/wordBE.png')
# plt.imshow(template), plt.show()
img = cv2.imread(img_folder_path + '/bothClose.png')
# plt.imshow(img), plt.show()

template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(template_gray, None)
kp2, des2 = orb.detectAndCompute(img_gray, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key = lambda x:x.distance)

print('matches: ', len(matches))

img2 = cv2.drawMatches(template_gray, kp1, img_gray, kp2, matches[:10], flags = cv2.DrawMatchesFlags_DEFAULT, outImg = None)

cv2.imshow('ORB', img2)

cv2.waitKey(0)
cv2.destroyAllWindows()