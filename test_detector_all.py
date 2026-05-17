import cv2
import numpy as np
from detector import VehicleDetector, DETECTOR_CONFIG
from preprocessing import preprocess_for_yolo

def main():
    img_path = r"D:\Traffic_AI\aa\frame_000001.png"
    img = cv2.imread(img_path)
    if img is None:
        print("Error: Could not load image.")
        return
        
    img_resized = cv2.resize(img, (1280, 720))
    img_rgb = preprocess_for_yolo(img_resized)
    
    # Use extremely sensitive detector configuration
    cfg = {
        **DETECTOR_CONFIG,
        "confidence_threshold": 0.01,
        "imgsz": 1280
    }
    detector = VehicleDetector(cfg)
    detections = detector.detect(img_rgb, 0)
    
    print(f"Total detections found at conf >= 0.01: {len(detections)}")
    for i, det in enumerate(detections):
        bbox = det["bbox"]
        cx = (bbox[0] + bbox[2]) // 2
        cy = (bbox[1] + bbox[3]) // 2
        print(f"[{i}] {det['class']} - conf: {det['confidence']:.4f} - bbox: {bbox} - centroid: ({cx}, {cy})")

if __name__ == "__main__":
    main()
