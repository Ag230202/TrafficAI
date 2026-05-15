"""
super_diagnose.py
-----------------
Runs YOLO at 0.05 confidence on a RAW frame (no CLAHE).
Finds out if the silver cars are detected AT ALL.
"""
import cv2
import numpy as np
from detector import VehicleDetector, DETECTOR_CONFIG
from lane_mapper import LaneMapper, LANE_CONFIG

FRAME_PATH = r"D:\Traffic_AI\aa\frame_000001.png"

def main():
    frame_bgr = cv2.imread(FRAME_PATH)
    if frame_bgr is None: return
    frame_bgr = cv2.resize(frame_bgr, (1280, 720))
    
    # Preprocess: No CLAHE, just standard contrast
    frame_clean = cv2.convertScaleAbs(frame_bgr, alpha=1.1, beta=10)
    frame_rgb = cv2.cvtColor(frame_clean, cv2.COLOR_BGR2RGB)
    
    # ── DETECT EVERYTHING ──
    # Low threshold, high resolution
    detector_cfg = {**DETECTOR_CONFIG, "confidence_threshold": 0.05, "imgsz": 1280}
    detector = VehicleDetector(detector_cfg)
    detections = detector.detect(frame_rgb, 1)
    
    lane_mapper = LaneMapper(LANE_CONFIG)
    
    print(f"\nDETECTIONS FOUND: {len(detections)}")
    for i, det in enumerate(detections):
        assigned = lane_mapper.assign_lane(det["bbox"])
        print(f"  #{i:<2} {det['class']:<8} conf={det['confidence']:.2f}  lane={str(assigned)}")

    output = frame_bgr.copy()
    # Draw ALL detections
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]
        assigned = lane_mapper.assign_lane(det["bbox"])
        color = (0, 255, 0) if assigned else (0, 0, 255)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.putText(output, f"#{i} {det['confidence']:.2f}", (x1, y1-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    cv2.imwrite(r"D:\Traffic_AI\super_diagnostic.jpg", output)
    print(f"\nSaved to: D:\\Traffic_AI\\super_diagnostic.jpg")

if __name__ == "__main__":
    main()
