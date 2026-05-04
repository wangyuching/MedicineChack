from ultralytics import YOLO
import os
import cv2
import numpy as np

model = YOLO("best.pt", task="obb") #best.float32.tflite, best.onnx
current_folder = os.path.dirname(os.path.abspath(__file__))
img = cv2.imread(os.path.join(current_folder,"image", "original0.png"))
# print(type(img))
results = model(img)

target_cls = 4

for r in results:
    classes = r.obb.cls
    # boxes = r.obb.xywhr
    boxes = r.obb.xyxyxyxy

    mask = (classes == target_cls)

    filter_boxes = boxes[mask]

    if len(filter_boxes) > 0:
        print(f"Class {target_cls} has {len(filter_boxes)} objects")
        # print(filter_boxes)
    else:
        print(f"There's no objects for Class {target_cls}")

    for box in filter_boxes:
        points = box.numpy().astype(np.int32)
        # print(points)
        cv2.polylines(img, [points], isClosed=True, color=(0, 0, 255), thickness=2)
cv2.imshow("obb draw point", img)
cv2.imwrite("poly.png",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# cap = cv2.VideoCapture(1)

# while cap.isOpened():

#     ok, frame = cap.read()
#     if (not ok) | (frame is None):    ""
#         print("usb pull out and in again...")
#         break
#     else:
#         results = model(frame)
#         for r in results:
#             if r.obb is not None:
#                 print(r.obb.xyxyxyxy)

#         annotated_frame = results[0].plot()
#         cv2.imshow("YOLO26 OBB Streaming", annotated_frame)

#         key = cv2.waitKey(1) & 0xFF
#         if key == ord("q") or key == ord("Q"):
#             break

# cap.release()
# cv2.destroyAllWindows()