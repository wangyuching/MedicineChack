# Dilstion - erosion = border
import cv2
import numpy as np

cap = cv2.VideoCapture(1)
kernel = np.ones((11, 11), np.uint8)
dilation_kernel = np.ones((5, 5), np.uint8)
erosion_kernel = np.ones((5, 5), np.uint8)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel) 
    dilation_img = cv2.dilate(closing, dilation_kernel, iterations=1)
    eroded_image = cv2.erode(closing, erosion_kernel, iterations=1)
    morphology_img = cv2.subtract(dilation_img, eroded_image)
        
    cv2.imshow('Frame', frame)
    cv2.imshow('Gray', gray)
    cv2.imshow('Morphology', morphology_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()