import time
import os
import numpy as np
import cv2
from ultralytics import YOLO

# 定義函數，用於獲取指定類別的 OBB (Oriented Bounding Box) 框
def get_target_obb(results, target_cls):
    filtered_boxes = []
    # 類別名稱
    name = ["bedtime_Word", "lid_close", "lid_hinge", "lid_open", "pill_box"]

    # 遍歷所有偵測結果
    for r in results:
        classes = r.obb.cls
        boxes = r.obb.xywhr #[center_x, center_y, width, height, rotation_radians]

        # 根據 target_cls 進行篩選
        mask = (classes == target_cls)
        target_boxes = boxes[mask]

        if len(target_boxes) > 0:
            print(f"Class {target_cls} ({name[target_cls]}) has {len(target_boxes)} objects")
        else:
            print(f"There's no objects for Class {target_cls}")
        
        # 將篩選出的框加入 filtered_boxes
        for box in target_boxes:
            filtered_boxes.append(box.numpy()) #.astype(np.int32)
        
    return filtered_boxes

# 定義函數，用於在影像上繪製 OBB 框
def draw_target_obb(image, boxes, color, thickness=2):
    output_img = image.copy()
    for box in boxes:
        x, y, w, h, r = box

        angle = np.degrees(r)
        # 建立旋轉矩陣
        rect = ((x, y), (w, h), angle)

        # 獲取旋轉矩陣的四個頂點
        points = cv2.boxPoints(rect)
        points = np.int32(points)
    
        # 繪製 OBB 框
        cv2.polylines(output_img, [points], isClosed=True, color=color, thickness=thickness)

    return output_img

# 定義函數，用於將 OBB 框沿著指定軸（w 或 h）分割成 num_splits 個子塊
def split_obb(obb_xywhr, axis='w', num_splits=4):
    xc, yc, w, h, r = obb_xywhr
    
    # 計算子塊的新尺寸
    new_w = w / num_splits if axis == 'w' else w
    new_h = h / num_splits if axis == 'h' else h
    
    sub_obbs = []
    
    # 計算每個子塊中心在局部坐標系下的偏移
    steps = np.linspace(-0.5 + 1/(2*num_splits), 0.5 - 1/(2*num_splits), num_splits)
    
    for step in steps:
        if axis == 'w':
            dx, dy = step * w, 0
        else:
            dx, dy = 0, step * h
            
        # 旋轉矩陣變換
        new_x = xc + dx * np.cos(r) - dy * np.sin(r)
        new_y = yc + dx * np.sin(r) + dy * np.cos(r)
        
        sub_obbs.append([new_x, new_y, new_w, new_h, r])
        
    return sub_obbs

# 定義函數，用於在指定子塊中檢測是否含有藥丸
def check_pill_in_split_box(frame, box, hsv_lower, hsv_upper, threshold=0.1):
    xc, yc, w, h, r = box

    # 建立旋轉矩陣
    M = cv2.getRotationMatrix2D((xc, yc), np.degrees(r), 1)
    # 進行仿射變換
    rotated = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
    # 裁切指定區域
    crop = cv2.getRectSubPix(rotated, (int(w), int(h)), (xc, yc))

    if (crop is None) or (crop.size == 0):
        return False, np.zeros((10, 10), dtype=np.uint8)
    
    # 將裁切區域轉換為 HSV 顏色空間
    hsv_img = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # 進行顏色篩選
    mask = cv2.inRange(hsv_img, hsv_lower, hsv_upper)

    # 計算篩選出的白色像素點數量
    white_pixels = cv2.countNonZero(mask)
    total_pixels = w * h
    ratio = white_pixels / total_pixels

    has_pill = ratio > threshold
    print(f"Box {i}: Pill Ratio = {ratio:.2%}")

    return has_pill, mask

# 載入 YOLO 模型
model = YOLO("best.pt", task="obb") #best.float32.tflite, best.onnx
# 開啟網路攝影機
cap = cv2.VideoCapture(1)

HSV_LOWER = np.array([0, 0, 255])
HSV_UPPER = np.array([179, 255, 255])

# 類別名稱
name = ["bedtime_Word", "lid_close", "lid_hinge", "lid_open", "pill_box"]

while cap.isOpened():
    ok, frame = cap.read()
    if (not ok) | (frame is None):    
        print("usb pull out and in again...")
        break
    else:
        # 縮放影像
        frame = cv2.resize(frame, (0, 0), fx=0.7, fy=0.7)
        # 進行偵測
        results = model(frame)
        # 在影像上標註偵測結果
        annotated_frame = results[0].plot()

        # 獲取藥丸盒 OBB 框
        pill_boxes = get_target_obb(results, target_cls=4)
        if pill_boxes:
            # 獲取關閉蓋子和打開蓋子 OBB 框
            lid_close = get_target_obb(results, target_cls=1)
            lid_open = get_target_obb(results, target_cls=3)
            # 判斷是否偵測到至少 4 個蓋子（關閉或打開）
            if (len(lid_close) + len(lid_open)) >= 4: 
                split_result_img = frame.copy()
                
                # 遍歷藥丸盒框
                for box in pill_boxes:
                    w, h = box[2], box[3]
                    # 決定分割軸
                    split_axis = "w" if w > h else "h"
                    # 將藥丸盒框分割成子塊
                    sub_boxes = split_obb(box, split_axis, num_splits=4)

                    masks_to_show = []
                    # 遍歷子塊，檢測是否含有藥丸
                    for i, sub_box in enumerate(sub_boxes):
                        has_pill, mask = check_pill_in_split_box(split_result_img, sub_box, HSV_LOWER, HSV_UPPER)
                        resized_mask = cv2.resize(mask, (100, 100))
                        masks_to_show.append(resized_mask)

                        color = (0, 255, 0) if has_pill else (0, 0, 255)
                        label = "Full" if has_pill else "Empty"
                        # 在影像上繪製分割後的框和文字
                        split_result_img = draw_target_obb(split_result_img, sub_boxes, (0, 255, 0), thickness=1)
                        cv2.putText(split_result_img, f"#{i}:{label}",
                                    (int(sub_box[0]-15), int(sub_box[1])),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    # 顯示子塊的 HSV mask
                    if len(masks_to_show) >= 4:
                        all_masks = cv2.hconcat(masks_to_show)
                        cv2.imshow("every HSV mask", all_masks)

                # 顯示分割後的結果
                cv2.imshow("split", split_result_img)
            else:
                print(f"{len(lid_close)} lid_close + {len(lid_open)} lid_open < 4 ")
                cv2.imshow("split", frame)

        else:
            print("Cant find object pill_box.")

        # 顯示標註偵測結果的影像
        cv2.imshow("YOLO26 OBB Streaming", annotated_frame)
        
        key = cv2.waitKey(1) & 0xFF
        # 按下 q 或 Q 離開
        if key == ord("q") or key == ord("Q"):
            break
        # 按下 p 或 P 暫停
        elif key == ord("p") or key == ord("P"):
            cv2.waitKey()

cap.release()
cv2.destroyAllWindows()