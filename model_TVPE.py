from  ultralytics import YOLO

if __name__ == "__main__":
    #train
    # model = YOLO("yolo26s-obb.pt")
    # results = model.train(
        # data="./data.yaml",
        # epochs=100,
        # batch=16,
        # imgsz=640,
        # device=0,
        # workers=0,
        # project="Yolo26s_OBB_Train",
        # name="d0427_v1_e100_b16_i640"
    # )

    #val
    # model = YOLO("./runs/obb/Yolo26s_OBB_Train/d0427_v1_e100_b16_i640/weights/best.pt")
    # results = model.val(
    #     data="./data.yaml",
    #     device=0,
    #     workers=0,
    #     project="Yolo26s_OBB_Val",
    #     name="d0427_v1_b16_i640"
    # )

    #predict
    # model = YOLO("./runs/obb/Yolo26s_OBB_Train/d0427_v1_e100_b16_i640/weights/best.pt")
    # results = model.predict(
    #     source=1,
    #     stream=True,
    #     show=True,
    #     # device=0,
    #     # workers=0,
    # )
    # for result in results:
    #     obbs = result.obb
    #     classes = result.obb.cls
    #     print(result.obb)

        # source="./data/validation/images",
        # device=0,
        # workers=0,
        # save=True,
        # project="Yolo26s_OBB_Predict",
        # name="d0427_v1_b16_i640"

    #export
    model = YOLO("best.pt", task="obb")
    model.export(format="tflite")