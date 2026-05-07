import time
import os
import numpy as np
import cv2
from ultralytics import YOLO

def get_target_obb(results, target_cls):
    filtered_boxes = []
    for r in results:
        classes = r.obb.cls
        boxes = r.obb.xywhr #[center_x, center_y, width, height, rotation_radians]

        mask = (classes == target_cls)
        target_boxes = boxes[mask]

        if len(target_boxes) > 0:
            print(f"Class {target_cls} has {len(target_boxes)} objects")
        else:
            print(f"There's no objects for Class {target_cls}")
        
        for box in target_boxes:
            filtered_boxes.append(box.numpy()) #.astype(np.int32)
            
        return filtered_boxes

def draw_target_obb(image, boxes, color, thickness=2):
    output_img = image.copy()
    for box in boxes:
        x, y, w, h, r = box

        
        if w > h:
            orientation = "Horizontal"
            label_color = (0, 0, 255) #horizontal is red.
        else:
            orientation = "Vertical"
            label_color = (0, 255, 0) #vertical is green.
        
        print(f"Object at ({x:.1f}, {y:.1f}) is {orientation} (w={w:.1f}, h={h:.1f})")


        angle = np.degrees(r)
        rect = ((x, y), (w, h), angle)

        points = cv2.boxPoints(rect)
        points = np.int32(points)
    
        cv2.polylines(output_img, [points], isClosed=True, color=color, thickness=thickness)

        cv2.putText(output_img, orientation, (int(x), int(y)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2)

    return output_img

model = YOLO("best.pt", task="obb") #best.float32.tflite, best.onnx

cap = cv2.VideoCapture(1)

image_folder = "image"
if not os.path.exists(image_folder):
    os.makedirs(image_folder)
    print(f"Create folder {image_folder} success")
image_number = 0

target_cls = 4

while cap.isOpened():

    t_start = time.perf_counter()

    ok, frame = cap.read()
    if (not ok) | (frame is None):    
        print("usb pull out and in again...")
        break
    else:
        results = model(frame)
        annotated_frame = results[0].plot()

        target_boxes = get_target_obb(results, target_cls)
        if target_boxes:
            annotated_frame = draw_target_obb(annotated_frame, target_boxes, (255, 255, 255))

        cv2.imshow("YOLO26 OBB Streaming", annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == ord("Q"):
            break
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