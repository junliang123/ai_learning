from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="yolo_dataset/data.yaml",
    epochs=1,
    imgsz=640,
    batch=8,
    workers=0,
    name="helmet_smoke",    
)