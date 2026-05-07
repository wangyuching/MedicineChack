from ultralytics import YOLO
import os
import cv2
import numpy as np

def load_model(model_path, task="obb"):
    return YOLO(model_path, task=task)

def get_target_obb(results, target_cls):
    filtered_boxes = []
    for r in results:
        classes = r.obb.cls
        boxes = r.obb.xywhr #[center_x, center_y, width, height, rotation_radians]

        mask = (classes == target_cls)
        target_boxes = boxes[mask]

        if len(target_boxes) > 0:
            print(f"Class {target_cls} has {len(target_boxes)} objects")
        else:
            print(f"There's no objects for Class {target_cls}")
        
        for box in target_boxes:
            filtered_boxes.append(box.numpy()) #.astype(np.int32)
            
        return filtered_boxes

def draw_target_obb(image, boxes, color, thickness=2):
    output_img = image.copy()
    for box in boxes:
        x, y, w, h, r = box
        angle = np.degrees(r)
        rect = ((x, y), (w, h), angle)

        points = cv2.boxPoints(rect)
        points = np.int32(points)
    
        cv2.polylines(output_img, [points], isClosed=True, color=color, thickness=thickness)

    return output_img

if __name__ == "__main__":
    current_folder = os.path.dirname(os.path.abspath(__file__))
    img = cv2.imread(os.path.join(current_folder,"image", "original.png"))

    if img is None:
        print("ERRERRRRRRR")

    else:
        model = load_model("best.pt")
        results = model(img)
        target_cls = 4
        boxes = get_target_obb(results, target_cls)

        if boxes:
            print(f"Class {target_cls} has {len(boxes)} objects.")
            result_img = draw_target_obb(img, boxes, (0, 0, 255))
            cv2.imshow("original", img)
            cv2.imshow("poly", result_img)
        else:
            print("ERROR")


cv2.waitKey(0)
cv2.destroyAllWindows()