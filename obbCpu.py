import time
import os
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


model = YOLO("best.pt", task="obb") #best.float32.tflite, best.onnx
cap = cv2.VideoCapture(1)

image_folder = "image"
if not os.path.exists(image_folder):
    os.makedirs(image_folder)
    print(f"Create folder {image_folder} success")
image_number = 0

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
                    split_result_img = draw_target_obb(split_result_img, sub_boxes, (0, 255, 0), thickness=1)
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
        elif key == ord("s") or key == ord("S"):
            original_image_name = os.path.join(image_folder,f"original{image_number}.png")
            cv2.imwrite(original_image_name, frame)
            print(f"Save {original_image_name} success")

            annotated_image_name = os.path.join(image_folder,f"annotated{image_number}.png")
            cv2.imwrite(annotated_image_name, annotated_frame)
            print(f"Save {annotated_image_name} success")
            
            image_number += 1

cap.release()
cv2.destroyAllWindows()