import cv2
import os
import numpy as np

def empty(v):
    pass

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

cv2.namedWindow('trackbar')
cv2.resizeWindow('trackbar', 500, 350)

# 建立六個軌跡條來調整HSV的範圍(名稱, 視窗名稱, 預設值, 最大值, 回調函數)
# hue:顏色，saturation:鮮豔度，value:亮度
cv2.createTrackbar('Hue_Min', 'trackbar', 0, 179, empty)
cv2.createTrackbar('Hue_Max', 'trackbar', 179, 179, empty)
cv2.createTrackbar('Sat_Min', 'trackbar', 0, 255, empty)
cv2.createTrackbar('Sat_Max', 'trackbar', 255, 255, empty)
cv2.createTrackbar('Val_Min', 'trackbar', 0, 255, empty)
cv2.createTrackbar('Val_Max', 'trackbar', 255, 255, empty)

while True:
    ret, img = cap.read()
    if not ret:
        print("Can't fing th cpmera")
        break
    img = cv2.resize(img, (300, 300))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

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

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()