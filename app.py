import os
import cv2
import time as t
import numpy as np
from db import PillManager
from ultralytics import YOLO
from alotdef import (get_target_obb, 
                     draw_target_obb, 
                     split_obb, 
                     lid_connect_split_box, 
                     pillbox_head_tail, 
                     check_pill_in_split_box,
                     draw_slot_states)

db_manager = PillManager()

def save_frame(frame, current_slots_data, tracker, duration):
    current_time = t.time()

    if "last_states" not in tracker:
        tracker["last_states"] = {i: {"lid": "Unknown", "Has_pill": False} for i in range(4)}
        tracker["open_start_time"] = None
        tracker["triggered_open"] = False
        tracker["pending_close_slots"] = {}

    current_opens = [idx for idx, data in current_slots_data.items() if data['lid'] == "Open"]

    if len(current_opens) > 0:
        if tracker["open_start_time"] is None:
            tracker["open_start_time"] = current_time
            tracker["triggered_open"] = False

        elif not tracker["triggered_open"]:
            elapsed_open = current_time - tracker["open_start_time"]
            if elapsed_open > duration:
                timestamp = t.strftime("%Y%m%d_%H%M%S")
                slot_details = [f"slot{i}_{'Full' if current_slots_data[i]['Has_pill'] else 'Empty'}" for i in current_opens]
                filename = f"saved_slots/{timestamp}_open_{'_'.join(slot_details)}.jpg"
                cv2.imwrite(filename, frame)
                db_manager.insert_pill_data(current_slots_data, frame)
                tracker["triggered_open"] = True
    else:
        tracker["open_start_time"] = None
        tracker["triggered_open"] = False
    
    triggered_close_events = False

    for i in range(4):
        last_lid = tracker["last_states"][i]["lid"]
        last_pill = tracker["last_states"][i]["Has_pill"]
        current_lid = current_slots_data[i]["lid"]

        was_open_and_full = (last_lid == "Open" and last_pill is True)
        now_close_or_missing = (current_lid in ["Close", "Missing"])

        if was_open_and_full and now_close_or_missing:
            if i not in tracker["pending_close_slots"]:
                tracker["pending_close_slots"][i] = current_time

        if current_lid == "Open" and (i in tracker["pending_close_slots"]):
            del tracker["pending_close_slots"][i]
        
        if now_close_or_missing and (i in tracker["pending_close_slots"]):
            elapsed_close = current_time - tracker["pending_close_slots"][i]
            if elapsed_close > duration:
                triggered_close_events = True
                del tracker["pending_close_slots"][i]

    if triggered_close_events:
        timestamp = t.strftime("%Y%m%d_%H%M%S")
        slot_details = [f"slot{i}_{'Full' if current_slots_data[i]['Has_pill'] else 'Empty'}" for i in range(4)]
        filename = f"saved_slots/{timestamp}_close_{'_'.join(slot_details)}.jpg"
        cv2.imwrite(filename, frame)
        db_manager.insert_pill_data(current_slots_data, frame)

    for i in range(4):
        tracker["last_states"][i] = {
            "lid": current_slots_data[i]["lid"], 
            "Has_pill": current_slots_data[i]["Has_pill"]
        }

model = YOLO("best.pt", task="obb") #best.float32.tflite, best.onnx
cap = cv2.VideoCapture(1)

HSV_LOWER = np.array([0, 0, 255])
HSV_UPPER = np.array([179, 255, 255])

reverse_state = False
has_once_detect_bedtime_word = False

saved_slots = "saved_slots"
if not os.path.exists(saved_slots):
    os.makedirs(saved_slots)

pill_tracker = {}

while cap.isOpened():
    ok, frame = cap.read()
    if (not ok) | (frame is None):    
        print("usb pull out and in again...")
        break
    else:
        frame = cv2.resize(frame, (640, 480))

        results = model(frame)
        annotated_frame = results[0].plot()
        annotated_frame = cv2.resize(annotated_frame, (640, 480))

        pill_detect_frame = frame.copy()

        bedtime_word = get_target_obb(results, target_cls=0)
        pill_boxes = get_target_obb(results, target_cls=4)
        if pill_boxes:
            lid_close = get_target_obb(results, target_cls=1)
            lid_open = get_target_obb(results, target_cls=3)
            all_lids = []
            for ls in lid_close: all_lids.append({'box': ls, 'state': 'Close'})
            for lo in lid_open: all_lids.append({'box': lo, 'state': 'Open'})

            if len(all_lids) > 0: 
                for pb in pill_boxes:
                    if bedtime_word:
                        current_res = pillbox_head_tail(bedtime_word, pb)
                        reverse_state = current_res
                        has_once_detect_bedtime_word = True
                        print("Direction updated by bedtime_word.")
                    elif has_once_detect_bedtime_word:
                        pass
                        print("Direction kept from last detection.")

                    w, h = pb[2], pb[3]
                    split_axis = "w" if w > h else "h"
                    sub_boxes = split_obb(pb, split_axis, num_splits=4, reverse=reverse_state)

                    slots_data = {i: {"lid" : "Missing", "Has_pill": False} for i in range(4)}
                    for lid in all_lids:
                        idx = lid_connect_split_box(lid['box'], sub_boxes)
                        if idx != -1:
                            slots_data[idx]['lid'] = lid['state']

                    for i, sub_box in enumerate(sub_boxes):
                        current_lid_state = slots_data[i]['lid']
                        if current_lid_state == "Open":
                            has_pill, mask = check_pill_in_split_box(pill_detect_frame, i, sub_box, HSV_LOWER, HSV_UPPER)
                            slots_data[i]['Has_pill'] = has_pill
                        else:
                            slots_data[i]['Has_pill'] = False
                        
                        if i == 3:
                            cv2.putText(pill_detect_frame, "TAIL", (int(sub_box[0]), int(sub_box[1]-20)), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                        
                        draw_slot_states(pill_detect_frame, sub_box, i, slots_data[i])

                    save_frame(
                        frame=pill_detect_frame,
                        current_slots_data=slots_data,
                        tracker=pill_tracker,
                        duration=5.0,
                    )

            else:
                print("Cant find any lids.")
        else:
            print("Cant find object pill_box.")

        combined_frame = np.hstack((frame, annotated_frame, pill_detect_frame))
        final_view = cv2.resize(combined_frame, (0, 0), fx=0.7, fy=0.7)
        cv2.imshow("YOLO26 OBB Streaming", final_view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == ord("Q"):
            break
        elif key == ord("p") or key == ord("P"):
            cv2.waitKey()

cap.release()
cv2.destroyAllWindows()