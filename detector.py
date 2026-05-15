"""
detector.py
-----------
Advanced Vehicle Detection using YOLOv8-Medium and ByteTrack.
"""

import numpy as np
import cv2

DETECTOR_CONFIG = {
    "model_path": "yolov8m.pt",
    "confidence_threshold": 0.04, # Extreme sensitivity
    "device": "cpu",
    "imgsz": 1280,
    "track_buffer": 60,
    "track_low_thresh": 0.04,
    "iou": 0.9 # Allow boxes to overlap (Essential for dense traffic)
}

class VehicleDetector:
    def __init__(self, config: dict = None):
        from ultralytics import YOLO
        cfg = config or DETECTOR_CONFIG
        self.config = cfg
        self.model = YOLO(cfg.get("model_path", "yolov8m.pt"))
        self.device = cfg.get("device", "cpu")
        self.model.to(self.device)
        
        # COCO Vehicle Classes: 2=car, 3=motorcycle, 5=bus, 7=truck
        self.target_classes = [2, 3, 5, 7]
        self.class_map = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

    def detect(self, frame_rgb: np.ndarray, frame_index: int = 0) -> list:
        """
        Runs YOLOv8 tracking (ByteTrack) on the frame.
        """
        results = self.model.track(
            frame_rgb,
            persist=True,
            conf=self.config.get("confidence_threshold", 0.04),
            iou=self.config.get("iou", 0.9),
            imgsz=self.config.get("imgsz", 1280),
            classes=self.target_classes,
            verbose=False,
            device=self.device,
            tracker="bytetrack.yaml"
        )
        
        return self._parse_tracking_results(results[0])

    def _parse_tracking_results(self, result) -> list:
        detections = []
        if not result.boxes:
            return detections

        for box in result.boxes:
            coords = box.xyxy[0].cpu().numpy()
            track_id = int(box.id[0].item()) if box.id is not None else -1
            conf = float(box.conf[0].item())
            cls_id = int(box.cls[0].item())
            
            detections.append({
                "bbox": [int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])],
                "confidence": conf,
                "class": self.class_map.get(cls_id, "vehicle"),
                "id": track_id
            })
            
        return detections

    @property
    def class_names(self) -> dict:
        return self.class_map
