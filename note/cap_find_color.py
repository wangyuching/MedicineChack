import cv2
import numpy as np

def empty(v):
    pass

cap = cv2.VideoCapture(1)

# 建立控制面板
cv2.namedWindow('trackbar')
cv2.resizeWindow('trackbar', 500, 300)
cv2.createTrackbar('Hue_Min', 'trackbar', 0, 179, empty)
cv2.createTrackbar('Hue_Max', 'trackbar', 179, 179, empty)
cv2.createTrackbar('Sat_Min', 'trackbar', 0, 255, empty)
cv2.createTrackbar('Sat_Max', 'trackbar', 255, 255, empty)
cv2.createTrackbar('Val_Min', 'trackbar', 0, 255, empty)
cv2.createTrackbar('Val_Max', 'trackbar', 255, 255, empty)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break
    
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    kernel = np.ones((3, 3), np.uint8)

    # 讀取軌跡條數值
    h_min = cv2.getTrackbarPos('Hue_Min', 'trackbar')
    h_max = cv2.getTrackbarPos('Hue_Max', 'trackbar')
    s_min = cv2.getTrackbarPos('Sat_Min', 'trackbar')
    s_max = cv2.getTrackbarPos('Sat_Max', 'trackbar')
    v_min = cv2.getTrackbarPos('Val_Min', 'trackbar')
    v_max = cv2.getTrackbarPos('Val_Max', 'trackbar')
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])

    # --- 1. 原始影像處理 ---
    hsv_raw = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_raw = cv2.inRange(hsv_raw, lower, upper)
    mask_raw = cv2.morphologyEx(mask_raw, cv2.MORPH_CLOSE, kernel)
    res_raw = cv2.bitwise_and(img, img, mask=mask_raw)

    # --- 2. HE (Histogram Equalization) 處理 ---
    hsv_he = hsv_raw.copy()
    # 僅針對 V 頻道做均衡化
    hsv_he[:, :, 2] = cv2.equalizeHist(hsv_he[:, :, 2])
    mask_he = cv2.inRange(hsv_he, lower, upper)
    mask_he = cv2.morphologyEx(mask_he, cv2.MORPH_CLOSE, kernel)
    # 轉回 BGR 以便顯示正常的彩色結果
    img_he = cv2.cvtColor(hsv_he, cv2.COLOR_HSV2BGR)
    res_he = cv2.bitwise_and(img_he, img_he, mask=mask_he)

    # --- 3. CLAHE 處理 ---
    hsv_clahe = hsv_raw.copy()
    clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    hsv_clahe[:, :, 2] = clahe_obj.apply(hsv_clahe[:, :, 2])
    mask_clahe = cv2.inRange(hsv_clahe, lower, upper)
    mask_clahe = cv2.morphologyEx(mask_clahe, cv2.MORPH_CLOSE, kernel)
    img_clahe = cv2.cvtColor(hsv_clahe, cv2.COLOR_HSV2BGR)
    res_clahe = cv2.bitwise_and(img_clahe, img_clahe, mask=mask_clahe)

    # --- 視窗整合與顯示 ---
    # 將原始、HE、CLAHE 的結果橫向拼接 (方便對照)
    top_row = np.hstack((img, img_he, img_clahe))
    mid_row = cv2.cvtColor(np.hstack((mask_raw, mask_he, mask_clahe)), cv2.COLOR_GRAY2BGR)
    bot_row = np.hstack((res_raw, res_he, res_clahe))
    
    # 堆疊三列
    combined = np.vstack((top_row, mid_row, bot_row))
    
    cv2.imshow('Comparison (Left:Raw | Mid:HE | Right:CLAHE)', combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == ord("Q"):
        break
    elif key == ord("p") or key == ord("P"):
        cv2.waitKey()

cap.release()
cv2.destroyAllWindows()