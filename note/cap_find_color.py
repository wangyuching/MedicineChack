# 找顏色
import cv2
import numpy as np

def empty(v):
    pass

cap = cv2.VideoCapture(1)

cv2.namedWindow('trackbar')
cv2.resizeWindow('trackbar', 500, 350)

    # 建立六個軌跡條來調整HSV的範圍(名稱, 視窗名稱, 預設值, 最大值, 回調函數)
cv2.createTrackbar('Hue_Min', 'trackbar', 0, 179, empty)
cv2.createTrackbar('Hue_Max', 'trackbar', 179, 179, empty)
cv2.createTrackbar('Sat_Min', 'trackbar', 0, 255, empty)
cv2.createTrackbar('Sat_Max', 'trackbar', 255, 255, empty)
cv2.createTrackbar('Val_Min', 'trackbar', 0, 255, empty)
cv2.createTrackbar('Val_Max', 'trackbar', 255, 255, empty)

while cap.isOpened():
    oh, img = cap.read()
    if not oh:
        break
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    img_kernel = img.copy()


    # hsv:色調、飽和度、亮度
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

    kernel = np.ones((3, 3), np.uint8)
    open_kernel = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    open_results = cv2.bitwise_and(img_kernel, img_kernel, mask=open_kernel)
    
    close_kernel = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    close_results = cv2.bitwise_and(img_kernel, img_kernel, mask=close_kernel)

    cv2.imshow('img', img)
    cv2.imshow('hsv', hsv)
    cv2.imshow('mask', mask)
    cv2.imshow('result', result)
    cv2.imshow('open_kernel', open_kernel)
    cv2.imshow('open_results', open_results)
    cv2.imshow('close_kernel', close_kernel)
    cv2.imshow('close_results', close_results)


    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == ord("Q"):
        break
    elif key == ord("p") or key == ord("P"):
        cv2.waitKey()
    # elif key == ord("s") or key == ord("S"):
    #     original_image_name = os.path.join(image_folder,f"original{image_number}.png")
    #     cv2.imwrite(original_image_name, frame)
    #     print(f"Save {original_image_name} success")

    #     annotated_image_name = os.path.join(image_folder,f"annotated{image_number}.png")
    #     cv2.imwrite(annotated_image_name, annotated_frame)
    #     print(f"Save {annotated_image_name} success")
        
        # image_number += 1

cap.release()
cv2.destroyAllWindows()