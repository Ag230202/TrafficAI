import os
import cv2
import numpy as np
from detector import VehicleDetector, DETECTOR_CONFIG
from preprocessing import preprocess_for_yolo

def main():
    frames_dir = r"D:\Traffic_AI\aa"
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")
    filenames = sorted(f for f in os.listdir(frames_dir) if f.lower().endswith(valid_ext))
    
    # Run with a super low threshold to see what's actually there
    cfg = {
        **DETECTOR_CONFIG,
        "confidence_threshold": 0.01,
        "imgsz": 1280
    }
    detector = VehicleDetector(cfg)
    
    print("Scanning for detections on the right half of the image (x > 600)...")
    
    for idx, fname in enumerate(filenames):
        filepath = os.path.join(frames_dir, fname)
        frame_bgr = cv2.imread(filepath)
        if frame_bgr is None:
            continue
            
        frame_resized = cv2.resize(frame_bgr, (1280, 720))
        frame_rgb = preprocess_for_yolo(frame_resized)
        
        detections = detector.detect(frame_rgb, idx)
        
        right_dets = []
        for det in detections:
            bbox = det["bbox"]
            cx = (bbox[0] + bbox[2]) // 2
            cy = (bbox[1] + bbox[3]) // 2
            if cx > 600:
                right_dets.append({
                    "class": det["class"],
                    "conf": det["confidence"],
                    "centroid": (cx, cy),
                    "bbox": bbox
                })
                
        if right_dets:
            print(f"\n--- Frame {fname} ---")
            for rd in right_dets:
                print(f"  {rd['class']} | conf: {rd['conf']:.4f} | centroid: {rd['centroid']} | bbox: {rd['bbox']}")

if __name__ == "__main__":
    main()
