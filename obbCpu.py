import os
import cv2
import numpy as np
from ultralytics import YOLO
from matplotlib.pyplot import box
from alotdef import (get_target_obb, 
                     draw_target_obb, 
                     split_obb, 
                     lid_connect_split_box, 
                     pillbox_head_tail, 
                     check_pill_in_split_box,
                     draw_slot_states)

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
            all_lids = []
            for ls in lid_close: all_lids.append({'box': ls, 'state': 'Close'})
            for lo in lid_open: all_lids.append({'box': lo, 'state': 'Open'})

            if len(all_lids) >= 4:#> 0: 
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

                    slots_data = {i: {"lid" : "Missing", "Has_pill": False} for i in range(4)}
                    for lid in all_lids:
                        idx = lid_connect_split_box(lid['box'], sub_boxes)
                        if idx != -1:
                            slots_data[idx]['lid'] = lid['state']

                    masks_to_show = []
                    for i, sub_box in enumerate(sub_boxes):
                        current_lid_state = slots_data[i]['lid']
                        if current_lid_state == "Open":
                            has_pill, mask = check_pill_in_split_box(split_result_img, i, sub_box, HSV_LOWER, HSV_UPPER)
                            slots_data[i]['Has_pill'] = has_pill
                            resized_mask = cv2.resize(mask, (100, 100))
                            masks_to_show.append(resized_mask)
                        else:
                            slots_data[i]['Has_pill'] = False                 
                        
                        if i == 3:
                            cv2.putText(split_result_img, "TAIL", (int(sub_box[0]), int(sub_box[1]-20)), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                        
                        draw_slot_states(split_result_img, sub_box, i, slots_data[i])


                    if len(masks_to_show) > 0:
                        all_masks = cv2.hconcat(masks_to_show)
                        cv2.imshow("every HSV mask", all_masks)
                cv2.imshow("split_result_img", split_result_img)

            else:
                print(f"{len(lid_close)} lid_close + {len(lid_open)} lid_open < 4 ")
                cv2.imshow("split_result_img", frame)

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