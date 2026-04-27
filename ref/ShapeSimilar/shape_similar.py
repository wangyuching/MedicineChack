# 輪廓近似(多邊形印似 > 推測輪廓形狀)
import cv2
import os
import numpy as np

current_path = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(current_path, 'shape.png')
img = cv2.imread(img_path)
img = cv2.resize(img, (500, 500))
img_contours = img.copy()
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(img, 10, 100)
# 偵測輪廓:輪廓, 階層 = (邊緣圖片, 輪廓檢索模式, 輪廓近似方法)
contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

for cnt in contours:
    cv2.drawContours(img_contours, cnt, -1, (255, 0, 0), 2)
    area = cv2.contourArea(cnt)
    if area > 500:
        peri = cv2.arcLength(cnt, True)
        vertices = cv2.approxPolyDP(cnt, peri * 0.02, True) 
        corners = (len(vertices))
        left_top_x, lest_top_y, w, h = cv2.boundingRect(vertices)
        cv2.rectangle(img_contours, (left_top_x, lest_top_y), (left_top_x + w, lest_top_y + h), (0, 255, 0), 1)
        if corners == 3:
            cv2.putText(img_contours, 'Triangle', (left_top_x, lest_top_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        elif corners == 4:
            cv2.putText(img_contours, 'Retengle', (left_top_x, lest_top_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        elif corners == 5:
            cv2.putText(img_contours, 'Pentagon', (left_top_x, lest_top_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        elif corners >= 6:
            cv2.putText(img_contours, 'Circle', (left_top_x, lest_top_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


cv2.imshow('img', img)
cv2.imshow('canny', canny)
cv2.imshow('contours', img_contours)
cv2.waitKey(0)