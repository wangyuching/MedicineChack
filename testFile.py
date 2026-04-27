import os
import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

folder_path = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(folder_path, 'testPic', 'bothClose.png')
if not os.path.exists(img_path):
    print("!!!!!! Image doesn't exist !!!!!!")
    os._exit(1)

img = cv2.imread(img_path)

plt.imshow(img), plt.show()