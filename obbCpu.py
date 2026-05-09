import time
import os
from matplotlib.pyplot import box
import numpy as np
import cv2
from ultralytics import YOLO

def get_target_obb(results, target_cls):
    filtered_boxes = []
    name = ["bedtime_Word", "lid_close", "lid_hinge", "lid_open", "pill_box"]

    for r in results:
        classes = r.obb.cls
        boxes = r.obb.xywhr #[center_x, center_y, width, height, rotation_radians]

        mask = (classes == target_cls)
        target_boxes = boxes[mask]

        if len(target_boxes) > 0:
            print(f"Class {target_cls} ({name[target_cls]}) has {len(target_boxes)} objects")
        else:
            print(f"There's no objects for Class {target_cls}")
        
        for box in target_boxes:
            filtered_boxes.append(box.numpy()) #.astype(np.int32)
            
        return filtered_boxes

def draw_target_obb(image, boxes, color, thickness=2):
    output_img = image.copy()
    for box in boxes:
        x, y, w, h, r = box

        angle = np.degrees(r)
        rect = ((x, y), (w, h), angle)

        points = cv2.boxPoints(rect)
        points = np.int32(points)
    
        cv2.polylines(output_img, [points], isClosed=True, color=color, thickness=thickness)

    return output_img

def split_obb(obb_xywhr, axis='w', num_splits=4):
    xc, yc, w, h, r = obb_xywhr
    
    # 計算子塊的新尺寸
    new_w = w / num_splits if axis == 'w' else w
    new_h = h / num_splits if axis == 'h' else h
    
    sub_obbs = []
    
    # 計算每個子塊中心在局部坐標系下的偏移
    # 例如 4 等分，比例為 -3/8, -1/8, 1/8, 3/8
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

def check_pill_in_split_box(frame, box, hsv_lower, hsv_upper, threshold=0.08):
    xc, xy, w, h, r = box

    # rect = ((xc, xy), (w, h), np.degrees(r))
    M = cv2.getRotationMatrix2D((xc, xy), np.degrees(r), 1)
    rotated = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
    crop = cv2.getRectSubPix(rotated, (int(w), int(h)), (xc, xy))

    if (crop is None) or (crop.size == 0):
        return False, np.zeros((10, 10), dtype=np.unit32)

    hsv_img = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, hsv_lower, hsv_upper)

    white_pixels = cv2.countNonZero(mask)
    total_pixels = w * h
    ratio = white_pixels / total_pixels

    has_pill = ratio> threshold
    print(f"Box {i}: Pill Ratio = {ratio:.2%}")

    return has_pill, mask

model = YOLO("best.pt", task="obb") #best.float32.tflite, best.onnx
cap = cv2.VideoCapture(1)

HSV_LOWER = np.array([0, 71, 0])
HSV_UPPER = np.array([179, 242, 233])

while cap.isOpened():
    ok, frame = cap.read()
    if (not ok) | (frame is None):    
        print("usb pull out and in again...")
        break
    else:
        frame = cv2.resize(frame, (0, 0), fx=0.7, fy=0.7)
        results = model(frame)
        annotated_frame = results[0].plot()

        pill_boxes = get_target_obb(results, target_cls=4)
        if pill_boxes:
            lid_close = get_target_obb(results, target_cls=1)
            lid_open = get_target_obb(results, target_cls=3)
            if (len(lid_close) + len(lid_open)) >= 4: 
                split_result_img = frame.copy()
                
                for box in pill_boxes:
                    w, h = box[2], box[3]
                    split_axis = "w" if w > h else "h"
                    sub_boxes = split_obb(box, split_axis, num_splits=4)

                    masks_to_show = []
                    for i, sub_box in enumerate(sub_boxes):
                        has_pill, mask = check_pill_in_split_box(split_result_img, sub_box, HSV_LOWER, HSV_UPPER)
                        resized_mask = cv2.resize(mask, (100, 100))
                        masks_to_show.append(resized_mask)

                        color = (0, 255, 0) if has_pill else (0, 0, 255)
                        label = "Full" if has_pill else "Empty"
                        split_result_img = draw_target_obb(split_result_img, sub_boxes, (0, 255, 0), thickness=1)
                        cv2.putText(split_result_img, f"#{i}:{label}",
                                    (int(sub_box[0]-15), int(sub_box[1])),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    if len(masks_to_show) >= 4:
                        all_masks = cv2.hconcat(masks_to_show)
                        cv2.imshow("every HSV mask", all_masks)

                cv2.imshow("split", split_result_img)
            else:
                print(f"{len(lid_close)} lid_close + {len(lid_open)} lid_open < 4 ")
                cv2.imshow("split", frame)

        else:
            print("Cant find object pill_box.")

        cv2.imshow("YOLO26 OBB Streaming", annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == ord("Q"):
            break
        elif key == ord("p") or key == ord("P"):
            cv2.waitKey()

cap.release()
cv2.destroyAllWindows()