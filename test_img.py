from ultralytics import YOLO
import os
import cv2
import numpy as np

model = YOLO("best.pt", task="obb") #best.float32.tflite, best.onnx
current_folder = os.path.dirname(os.path.abspath(__file__))
img = cv2.imread(os.path.join(current_folder,"image", "original.png"))
results = model(img)

target_cls = 4

for r in results:
    classes = r.obb.cls
    boxes = r.obb.xyxyxyxy #rb[0] rt[1] lt[2] lb[3]
    mask = (classes == target_cls)
    filter_boxes = boxes[mask]

    if len(filter_boxes) > 0:
        print(f"Class {target_cls} has {len(filter_boxes)} objects")
    else:
        print(f"There's no objects for Class {target_cls}")

    for i, box in enumerate(filter_boxes):
        points = box.numpy().astype(np.int32)
        
        p0, p1, p2, p3 = points[0], points[1], points[2], points[3]
        for k in range(1, 4):
            ratio = k / 4.0
            points_start = np.round(p0 + (p1 - p0) * ratio).astype(np.int32)
            points_end = np.round(p3 + (p2 - p3) * ratio).astype(np.int32)

            cv2.line(img, points_start, points_end, color=(0,255,0), thickness=2)

cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

    # for i, box in enumerate(filter_boxes):
    #     points = box.numpy().astype(np.int32)
    #     print(points, end="\n\n")
    #     cv2.polylines(img, [points], isClosed=True, color=(0, 0, 255), thickness=2)


    # for i, box in enumerate(filter_boxes):
    #     points = box.numpy().astype(np.float32)

    #     width = int(np.linalg.norm(points[1] - points[2]))
    #     height = int(np.linalg.norm(points[1] - points[0]))

    #     disten_points = np.array([
    #         [width, height], [width, 0], [0, 0], [0, height]
    #         ], dtype=np.float32)

    #     matrix = cv2.getPerspectiveTransform(points, disten_points)
    #     cropped_img = cv2.warpPerspective(img, matrix, (width, height))
    #     w_step = width // 4
    #     sub_parts = []
    #     for i in range(4):
    #         sub_part = cropped_img[:, i*w_step : (i+1)*w_step]
    #         sub_parts.append(sub_part)

    #         cv2.imshow(f"Part{i}", sub_part)

        #     width = int(np.linalg.norm(points[1] - points[2]))
        # height = int(np.linalg.norm(points[1] - points[0]))

        # disten_points = np.array([
        #     [width, height], [width, 0], [0, 0], [0, height]
        #     ], dtype=np.float32)

        # matrix = cv2.getPerspectiveTransform(points, disten_points)
        # cropped_img = cv2.warpPerspective(img, matrix, (width, height))
        # w_step = width // 4
        # sub_parts = []
        # for i in range(4):
        #     sub_part = cropped_img[:, i*w_step : (i+1)*w_step]
        #     sub_parts.append(sub_part)

        #     # cv2.imshow(f"Part{i}", sub_part)
        #     cv2.polylines(img, sub_part, isClosed=True, color=(0, 0, 255), thickness=2)



        # width = int(np.linalg.norm(points[0] - points[1]))
        # height = int(np.linalg.norm(points[1] - points[2]))

        # if height > width:
        #     width, height = height, width
        #     points = points[[1, 2, 3, 0]]

        # disten_points = np.array([
        #     [width, height], [width, 0], [0, 0], [0, height]
        #     ], dtype=np.float32)

        # matrix = cv2.getPerspectiveTransform(points, disten_points)
        # cropped_img = cv2.warpPerspective(img, matrix, (int(width), int(height)))
        # w_step = width // 4
        # sub_parts = []
        # for i in range(4):
        #     sub_part = cropped_img[:, i*w_step : (i+1)*w_step]
        #     sub_parts.append(sub_part)

        #     cv2.imshow(f"Part{i}", sub_part)