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
    
    detector = VehicleDetector(DETECTOR_CONFIG)
    lane_mapper = LaneMapper(LANE_CONFIG)
    
    print(f"Analyzing {len(filenames)} frames...")
    
    for idx, fname in enumerate(filenames):
        filepath = os.path.join(frames_dir, fname)
        frame_bgr = cv2.imread(filepath)
        if frame_bgr is None:
            continue
            
        frame_resized = cv2.resize(frame_bgr, (1280, 720))
        frame_rgb = preprocess_for_yolo(frame_resized)
        
        detections = detector.detect(frame_rgb, idx)
        
        # Check corner detections: x < 250 or x > 1000 or y > 550
        corner_dets = []
        for det in detections:
            bbox = det["bbox"]
            x1, y1, x2, y2 = bbox
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            # Corner regions: left side (x < 350) or right/bottom side (x > 900)
            if cx < 350 or cx > 900 or cy > 550:
                assigned_lane = lane_mapper.assign_lane(bbox)
                corner_dets.append({
                    "class": det["class"],
                    "conf": det["confidence"],
                    "bbox": bbox,
                    "centroid": (cx, cy),
                    "lane": assigned_lane
                })
                
        if corner_dets:
            print(f"\n--- Frame: {fname} (Index: {idx}) ---")
            for cd in corner_dets:
                print(f"  {cd['class']} | conf: {cd['conf']:.2f} | centroid: {cd['centroid']} | bbox: {cd['bbox']} | lane: {cd['lane']}")

if __name__ == "__main__":
    main()
