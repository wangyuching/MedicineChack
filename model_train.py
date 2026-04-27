from  ultralytics import YOLO

model = YOLO("yolo26n-obb.pt")
results = model.train(
    data="./data.yaml",
    epochs=100,
    batch=8,
    imgsz=640,
    device=0,
    project="Yolo26n_OBB_Train",
    name="d0427_v1_e100_b8_i640"
)