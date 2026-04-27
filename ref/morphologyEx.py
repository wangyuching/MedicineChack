import cv2
import numpy as np

cap = cv2.VideoCapture(1)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((11, 11), np.uint8) #結構元素(kernel
    #開運算:除影像中的白色雜訊或線條。
    opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel) 
    #閉運算:填補黑色小洞。
    closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel) 
    #禮帽運算:顯示白線條。
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    #黑帽運算:顯示黑色點。
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)


    cv2.imshow('Camera', frame)
    cv2.imshow('Gray', gray)
    cv2.imshow('Opening', opening)
    cv2.imshow('Closing', closing)
    cv2.imshow('TopHat', tophat)
    cv2.imshow('BlackHat', blackhat)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()