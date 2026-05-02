import cv2
from ultralytics import YOLO

model = YOLO("best_float32.tflite", task="obb")

cap = cv2.VideoCapture(1)

while cap.isOpened():
    ok, frame = cap.read()
    if ok:
        results = model(frame)
        annotated_frame = results[0].plot()
        cv2.imshow("YOLO26 OBB Streaming", annotated_frame)
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()