import cv2
import numpy as np
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(1)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # convert the image into grayscale before doing histogram equalization
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # image equalization
    equalize_img = cv2.equalizeHist(gray_img)

    # create clahe image
    clahe = cv2.createCLAHE()
    clahe_img = clahe.apply(gray_img)

    # show image
    cv2.imshow("image", gray_img)
    cv2.imshow("equal_image", equalize_img)
    cv2.imshow("clahe_image", clahe_img)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()