from ultralytics import YOLO
import os
import cv2
import numpy as np

model = YOLO("best.pt", task="obb") #best.float32.tflite, best.onnx

cap = cv2.VideoCapture(1)

target_cls = 4

while cap.isOpened():

    ok, frame = cap.read()
    if (not ok) | (frame is None):
        print("usb pull out and in again...")
        break
    else:
        results = model(frame)
        poly_frame = frame.copy()
        for r in results:
            classes = r.obb.cls
            boxes = r.obb.xyxyxyxy

            mask = (classes == target_cls)

            filter_boxes = boxes[mask]

            if len(filter_boxes) > 0:
                print(f"Class {target_cls} has {len(filter_boxes)} objects")
            else:
                print(f"There's no objects for Class {target_cls}")

            for box in filter_boxes:
                points = box.numpy().astype(np.int32)
                cv2.polylines(poly_frame, [points], isClosed=True, color=(0, 0, 255), thickness=2)
        cv2.imshow("poly", poly_frame)

        annotated_frame = results[0].plot()
        cv2.imshow("annnotated", annotated_frame)

        cv2.imshow("original", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == ord("Q"):
            break
        elif key == ord("s") or key == ord("S"):
            cv2.imwrite("original.png", frame)
            cv2.imwrite("poly.png", poly_frame)
            cv2.imwrite("annotated.png", annotated_frame)
        elif key == ord("p") or key == ord("P"):
            cv2.waitKey()

cap.release()
cv2.destroyAllWindows()