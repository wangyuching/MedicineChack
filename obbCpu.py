import cv2
import time
import numpy as np
from ultralytics import YOLO

model = YOLO("best.pt", task="obb") #best.float32.tflite, best.onnx

cap = cv2.VideoCapture(1)

avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200

while cap.isOpened():

    t_start = time.perf_counter()

    ok, frame = cap.read()
    if (not ok) | (frame is None):    
        print("usb pull out and in again...")
        break
    else:
        results = model(frame)
        annotated_frame = results[0].plot()
        # Draw framerate
        cv2.putText(annotated_frame, f'FPS: {avg_frame_rate:0.2f}', (10,20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)

        cv2.imshow("YOLO26 OBB Streaming", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Calculate FPS for this frame
        t_stop = time.perf_counter()
        frame_rate_calc = float(1/(t_stop - t_start))

        # Append FPS result to frame_rate_buffer (for finding average FPS over multiple frames)
        if len(frame_rate_buffer) >= fps_avg_len:
            temp = frame_rate_buffer.pop(0)
            frame_rate_buffer.append(frame_rate_calc)
        else:
            frame_rate_buffer.append(frame_rate_calc)

        # Calculate average FPS for past frames
        avg_frame_rate = np.mean(frame_rate_buffer)

cap.release()
cv2.destroyAllWindows()