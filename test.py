from ultralytics import YOLO

# Load a model
source = "https://ultralytics.com/images/bus.jpg"
model = YOLO('yolov8n.pt')  # load an official model

# Predict with the model
results = model.predict(source, save=True, imgsz=320, conf=0.5)