import os
import cv2
import numpy as np
from ultralytics import YOLO
from matplotlib.pyplot import box
from alotdef import (get_target_obb, 
                     draw_target_obb, 
                     pillbox_head_tail, 
                     split_obb, 
                     check_pill_in_split_box)

model = YOLO("best.pt", task="obb") #best.float32.tflite, best.onnx
cap = cv2.VideoCapture(1)

HSV_LOWER = np.array([0, 0, 255])
HSV_UPPER = np.array([179, 255, 255])

reverse_state = False
has_once_detect_bedtime_word = False

while cap.isOpened():
    ok, frame = cap.read()
    if (not ok) | (frame is None):    
        print("usb pull out and in again...")
        break
    else:
        frame = cv2.resize(frame, (0, 0), fx=0.7, fy=0.7)
        results = model(frame)
        annotated_frame = results[0].plot()

        bedtime_word = get_target_obb(results, target_cls=0)
        pill_boxes = get_target_obb(results, target_cls=4)
        if pill_boxes:
            lid_close = get_target_obb(results, target_cls=1)
            lid_open = get_target_obb(results, target_cls=3)
            if (len(lid_close) + len(lid_open)) >= 4: 
                split_result_img = frame.copy()
                for box in pill_boxes:
                    if bedtime_word:
                        current_res = pillbox_head_tail(bedtime_word, box)
                        reverse_state = current_res
                        has_once_detect_bedtime_word = True
                        print("Direction updated by bedtime_word.")
                    elif has_once_detect_bedtime_word:
                        pass
                        print("Direction kept from last detection.")

                    w, h = box[2], box[3]
                    split_axis = "w" if w > h else "h"
                    sub_boxes = split_obb(box, split_axis, num_splits=4, reverse=reverse_state)

                    masks_to_show = []
                    for i, sub_box in enumerate(sub_boxes):
                        has_pill, mask = check_pill_in_split_box(split_result_img, i, sub_box, HSV_LOWER, HSV_UPPER)
                        resized_mask = cv2.resize(mask, (100, 100))
                        masks_to_show.append(resized_mask)

                        color = (0, 255, 0) if has_pill else (0, 0, 255)
                        label = "Full" if has_pill else "Empty"
                        split_result_img = draw_target_obb(split_result_img, sub_boxes, (0, 255, 0), thickness=1)
                        cv2.putText(split_result_img, f"#{i}:{label}",
                                    (int(sub_box[0]-15), int(sub_box[1])),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                        if i == 3:
                            cv2.putText(split_result_img, "TAIL", (int(sub_box[0]), int(sub_box[1]-20)), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                    if len(masks_to_show) >= 4:
                        all_masks = cv2.hconcat(masks_to_show)
                        cv2.imshow("every HSV mask", all_masks)

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

cap.release()
cv2.destroyAllWindows()