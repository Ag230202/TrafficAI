import cv2
import numpy as np
from detector import VehicleDetector, DETECTOR_CONFIG
from preprocessing import preprocess_for_yolo, apply_clahe, convert_bgr_to_rgb

def test_preprocess(name, frame_rgb, detector):
    detections = detector.detect(frame_rgb, 0)
    print(f"  [{name}] found {len(detections)} detections:")
    for i, det in enumerate(detections):
        bbox = det["bbox"]
        cx = (bbox[0] + bbox[2]) // 2
        cy = (bbox[1] + bbox[3]) // 2
        print(f"    - {det['class']} (conf={det['confidence']:.2f}) centroid=({cx}, {cy}) bbox={bbox}")

def main():
    img_path = r"D:\Traffic_AI\aa\frame_000001.png"
    img = cv2.imread(img_path)
    if img is None:
        print("Error: Could not load image.")
        return
        
    img_resized = cv2.resize(img, (1280, 720))
    
    # Set up detector with low confidence threshold
    cfg = {
        **DETECTOR_CONFIG,
        "confidence_threshold": 0.05,
        "imgsz": 1280
    }
    detector = VehicleDetector(cfg)
    
    print("=== Testing frame_000001.png ===")
    
    # 1. Raw BGR to RGB
    frame_raw = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    test_preprocess("RAW", frame_raw, detector)
    
    # 2. Pipeline Preprocess (Gamma + Subtle CLAHE)
    frame_pipe = preprocess_for_yolo(img_resized)
    test_preprocess("Shadow-Vision (Gamma + Subtle CLAHE)", frame_pipe, detector)
    
    # 3. CLAHE Only (from diagnose_lanes.py)
    frame_clahe = img_resized.copy()
    frame_clahe = apply_clahe(frame_clahe, clip_limit=2.0, tile_grid=(8, 8))
    frame_clahe = cv2.cvtColor(frame_clahe, cv2.COLOR_BGR2RGB)
    test_preprocess("CLAHE Only (clip_limit=2.0)", frame_clahe, detector)

if __name__ == "__main__":
    main()
