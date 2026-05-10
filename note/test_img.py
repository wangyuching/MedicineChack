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

        
        if w > h:
            orientation = "Horizontal"
            label_color = (0, 0, 255) #horizontal is red.
        else:
            orientation = "Vertical"
            label_color = (0, 255, 0) #vertical is green.
        
        print(f"Object at ({x:.1f}, {y:.1f}) is {orientation} (w={w:.1f}, h={h:.1f})")


        angle = np.degrees(r)
        rect = ((x, y), (w, h), angle)

        points = cv2.boxPoints(rect)
        points = np.int32(points)
    
        cv2.polylines(output_img, [points], isClosed=True, color=color, thickness=thickness)

        cv2.putText(output_img, orientation, (int(x), int(y)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2)

    return output_img

def split_obb(obb_xywhr, axis='w', num_splits=4):
    xc, yc, w, h, r = obb_xywhr
    
    # 計算子塊的新尺寸
    new_w = w / num_splits if axis == 'w' else w
    new_h = h / num_splits if axis == 'h' else h
    
    sub_obbs = []
    
    # 計算每個子塊中心在局部坐標系下的偏移
    # 例如 4 等分，比例為 -3/8, -1/8, 1/8, 3/8
    steps = np.linspace(-0.5 + 1/(2*num_splits), 0.5 - 1/(2*num_splits), num_splits)
    
    for step in steps:
        if axis == 'w':
            dx, dy = step * w, 0
        else:
            dx, dy = 0, step * h
            
        # 旋轉矩陣變換
        new_x = xc + dx * np.cos(r) - dy * np.sin(r)
        new_y = yc + dx * np.sin(r) + dy * np.cos(r)
        
        sub_obbs.append([new_x, new_y, new_w, new_h, r])
        
    return sub_obbs

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

            split_result_img = img.copy()
            for box in boxes:
                w, h = box[2], box[3]
                split_axis = "w" if w > h else "h"
                sub_boxes = split_obb(box, split_axis, num_splits=4)
                split_result_img = draw_target_obb(split_result_img, sub_boxes, (0, 255, 0), thickness=1)

            # result_img = draw_target_obb(img, boxes, (0, 0, 255))
            cv2.imshow("original", img)
            cv2.imshow("split", split_result_img)
        else:
            print("NOTHING")

cv2.waitKey(0)
cv2.destroyAllWindows()