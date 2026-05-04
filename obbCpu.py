import time
import os
import numpy as np
import cv2
from ultralytics import YOLO

model = YOLO("best.pt", task="obb") #best.float32.tflite, best.onnx

cap = cv2.VideoCapture(1)

image_folder = "image"
if not os.path.exists(image_folder):
    os.makedirs(image_folder)
    print(f"Create folder {image_folder} success")
image_number = 0

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
        cv2.putText(annotated_frame, f"FPS: {avg_frame_rate:0.2f}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)

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