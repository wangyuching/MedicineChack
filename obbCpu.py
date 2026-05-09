import time
import os
import numpy as np
import cv2
from ultralytics import YOLO

class ColorFinder:
    def __init__(self, window_name='Color Trackbar'):
        self.window_name = window_name
        cv2.namedWindow(self.window_name)
        w = int(500*0.7)
        h = int(350*0.7)
        cv2.resizeWindow(self.window_name, w, h)
        
        # 初始化控制條 (預設值設為 0-179, 0-255, 0-255 即顯示全彩)
        cv2.createTrackbar('Hue_Min', self.window_name, 0, 179, self._nothing)
        cv2.createTrackbar('Hue_Max', self.window_name, 179, 179, self._nothing)
        cv2.createTrackbar('Sat_Min', self.window_name, 0, 255, self._nothing)
        cv2.createTrackbar('Sat_Max', self.window_name, 255, 255, self._nothing)
        cv2.createTrackbar('Val_Min', self.window_name, 0, 255, self._nothing)
        cv2.createTrackbar('Val_Max', self.window_name, 255, 255, self._nothing)

    def _nothing(self, x):
        pass

    def get_mask_and_result(self, frame):
        # 轉換為 HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 取得目前控制條數值
        h_min = cv2.getTrackbarPos('Hue_Min', self.window_name)
        h_max = cv2.getTrackbarPos('Hue_Max', self.window_name)
        s_min = cv2.getTrackbarPos('Sat_Min', self.window_name)
        s_max = cv2.getTrackbarPos('Sat_Max', self.window_name)
        v_min = cv2.getTrackbarPos('Val_Min', self.window_name)
        v_max = cv2.getTrackbarPos('Val_Max', self.window_name)
        
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        
        # 產生遮罩與結果
        mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(frame, frame, mask=mask)
        
        return mask, result

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
color_finder = ColorFinder("Color Finder")

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

        mask, color_results = color_finder.get_mask_and_result(frame)

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
        cv2.imshow("hsv mask", mask)
        cv2.imshow("color filter results", color_results)

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