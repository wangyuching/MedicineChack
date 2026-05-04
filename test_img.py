from ultralytics import YOLO
import os
import cv2

model = YOLO("best.pt", task="obb") #best.float32.tflite, best.onnx
current_folder = os.path.dirname(os.path.abspath(__file__))
img = os.path.join(current_folder,"image", "original0.png")
print(type(img))
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
        points = box.numpy()
        print(points)
    

# # 矩形 (圖片, 左上角座標, 右下角座標, 顏色BGR, 線條粗細/填滿)
# cv2.rectangle(img, (50, 50), (250, 150), (0, 0, 255), 2)  


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