import cv2
from ultralytics import YOLO

img_path = r"E:\photos\selected\selected\frame_000042.png"
img = cv2.imread(img_path)
model = YOLO('yolov8n.pt')

results = model(img)
for r in results:
    for box in r.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        name = model.names[cls]
        if name in ['car', 'truck', 'bus', 'motorcycle']:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            area = (x2 - x1) * (y2 - y1)
            print(f"Vehicle: {name}, Conf: {conf:.2f}, Area: {area:.0f}, Center: ({(x1+x2)/2:.0f}, {(y1+y2)/2:.0f})")
