import cv2
import numpy as np

cap = cv2.VideoCapture(1)
kernel_20 = np.ones((20, 20), np.uint8) #結構元素(kernel
kernel_11 = np.ones((11, 11), np.uint8) #結構元素(kernel
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #閉運算:填補黑色小洞。
    
    closing_20 = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel_20) 
    closing_11 = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel_11) 
    cv2.imshow('Camera', frame)
    cv2.imshow('Gray', gray)
    cv2.imshow('Closing_20', closing_20)
    cv2.imshow('Closing_11', closing_11)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()