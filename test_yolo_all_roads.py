import os
import cv2
import numpy as np
from detector import VehicleDetector, DETECTOR_CONFIG
from lane_mapper import LaneMapper, LANE_CONFIG
from preprocessing import preprocess_for_yolo

def main():
    frames_dir = r"D:\Traffic_AI\aa"
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")
    filenames = sorted(f for f in os.listdir(frames_dir) if f.lower().endswith(valid_ext))
    
    # Let's run with confidence threshold 0.01 to see if we capture the vehicles at all
    cfg = {
        **DETECTOR_CONFIG,
        "confidence_threshold": 0.01,
        "imgsz": 1280
    }
    detector = VehicleDetector(cfg)
    lane_mapper = LaneMapper(LANE_CONFIG)
    
    print(f"Running detection on {len(filenames)} frames...")
    
    lane_counts_all = {lane: 0 for lane in LANE_CONFIG}
    lane_counts_all["None"] = 0
    
    for idx, fname in enumerate(filenames):
        filepath = os.path.join(frames_dir, fname)
        frame_bgr = cv2.imread(filepath)
        if frame_bgr is None:
            continue
            
        frame_resized = cv2.resize(frame_bgr, (1280, 720))
        frame_rgb = preprocess_for_yolo(frame_resized)
        
        detections = detector.detect(frame_rgb, idx)
        
        print(f"\n--- Frame {fname} ({len(detections)} detections) ---")
        for det in detections:
            bbox = det["bbox"]
            cx = (bbox[0] + bbox[2]) // 2
            cy = (bbox[1] + bbox[3]) // 2
            assigned_lane = lane_mapper.assign_lane(bbox)
            
            lane_counts_all[str(assigned_lane)] += 1
            print(f"  {det['class']} | conf: {det['confidence']:.2f} | centroid: ({cx}, {cy}) | bbox: {bbox} | lane: {assigned_lane}")
            
    print("\n================== TOTAL DETECTIONS PER LANE ==================")
    for lane, count in lane_counts_all.items():
        print(f"  {lane}: {count}")

if __name__ == "__main__":
    main()
