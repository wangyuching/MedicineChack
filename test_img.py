from ultralytics import YOLO
import os
import cv2
import numpy as np

model = YOLO("best.pt", task="obb") #best.float32.tflite, best.onnx
current_folder = os.path.dirname(os.path.abspath(__file__))
img = cv2.imread(os.path.join(current_folder,"image", "original0.png"))
cv2.imshow("original", img)
# print(type(img))
results = model(img)

target_cls = 4

for r in results:
    classes = r.obb.cls
    boxes = r.obb.xyxyxyxy #rb[0] rt[1] lt[2] lb[3]

    mask = (classes == target_cls)

    filter_boxes = boxes[mask]

    if len(filter_boxes) > 0:
        print(f"Class {target_cls} has {len(filter_boxes)} objects")
    else:
        print(f"There's no objects for Class {target_cls}")

    for box in filter_boxes:
        points = box.numpy().astype(np.int32)
        print(points, end="\n\n")
        cv2.polylines(img, [points], isClosed=True, color=(0, 0, 255), thickness=2)
        
        # print(points[0][0])
        roi = img[(points[2][1]):(points[0][1]), (points[2][0]):(points[0][0])]
        cv2.imshow("roi", roi)


    

cv2.imshow("poly", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

