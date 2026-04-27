from  ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo26n-obb.pt")
    results = model.train(
    data="./data.yaml",
    epochs=100,
    batch=16,
    imgsz=640,
    device=0,
    workers=0,
    project="Yolo26n_OBB_Train",
    name="d0427_v1_e100_b16_i640"
    )