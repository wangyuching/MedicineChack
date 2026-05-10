# 找顏色
import os
import cv2
import numpy as np

def empty(v):
    pass

current_folder = os.path.dirname(os.path.abspath(__file__))
base_root = os.path.dirname(current_folder)
img = cv2.imread(os.path.join(base_root,"image", "clahe.png"))
img = cv2.resize(img, (0, 0), fx=0.7, fy=0.7)

cv2.namedWindow('trackbar')
cv2.resizeWindow('trackbar', 500, 350)

# 建立六個軌跡條來調整HSV的範圍(名稱, 視窗名稱, 預設值, 最大值, 回調函數)
cv2.createTrackbar('Hue_Min', 'trackbar', 0, 179, empty)
cv2.createTrackbar('Hue_Max', 'trackbar', 179, 179, empty)
cv2.createTrackbar('Sat_Min', 'trackbar', 0, 255, empty)
cv2.createTrackbar('Sat_Max', 'trackbar', 255, 255, empty)
cv2.createTrackbar('Val_Min', 'trackbar', 0, 255, empty)
cv2.createTrackbar('Val_Max', 'trackbar', 255, 255, empty)

# hsv:色調、飽和度、亮度
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
while True:
    h_min = cv2.getTrackbarPos('Hue_Min', 'trackbar')
    h_max = cv2.getTrackbarPos('Hue_Max', 'trackbar')
    s_min = cv2.getTrackbarPos('Sat_Min', 'trackbar')
    s_max = cv2.getTrackbarPos('Sat_Max', 'trackbar')
    v_min = cv2.getTrackbarPos('Val_Min', 'trackbar')
    v_max = cv2.getTrackbarPos('Val_Max', 'trackbar')
    print(h_min, h_max, s_min, s_max, v_min, v_max)

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])

    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(img, img, mask=mask)

    cv2.imshow('img', img)
    cv2.imshow('hsv', hsv)
    cv2.imshow('mask', mask)
    cv2.imshow('result', result)
    cv2.waitKey(1)