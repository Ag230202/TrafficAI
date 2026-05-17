import cv2
import numpy as np
from detector import VehicleDetector, DETECTOR_CONFIG
from lane_mapper import LaneMapper, LANE_CONFIG
from preprocessing import preprocess_for_yolo

def main():
    img_path = r"D:\Traffic_AI\aa\frame_000001.png"
    img = cv2.imread(img_path)
    if img is None:
        print("Error: Could not load image.")
        return
        
    img_resized = cv2.resize(img, (1280, 720))
    img_rgb = preprocess_for_yolo(img_resized)
    
    detector = VehicleDetector(DETECTOR_CONFIG)
    detections = detector.detect(img_rgb, 0)
    
    print("--- Polygons ---")
    for name, poly in LANE_CONFIG.items():
        print(f"  {name}: {poly}")
        
    print("\n--- Detections and Polygon mapping ---")
    for i, det in enumerate(detections):
        bbox = det["bbox"]
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        point = (cx, cy)
        
        print(f"\n[{i}] {det['class']} (conf={det['confidence']:.2f}) bbox={bbox} centroid=({cx}, {cy})")
        
        # Check inside
        inside_lane = None
        for name, poly in LANE_CONFIG.items():
            poly_np = np.array(poly, dtype=np.int32)
            inside = cv2.pointPolygonTest(poly_np, point, False)
            if inside >= 0:
                inside_lane = name
                break
                
        if inside_lane:
            print(f"  -> DIRECTLY INSIDE: {inside_lane}")
        else:
            print("  -> OUTSIDE all polygons! Calculating distances:")
            for name, poly in LANE_CONFIG.items():
                poly_np = np.array(poly, dtype=np.int32)
                dist = cv2.pointPolygonTest(poly_np, point, True)
                print(f"     dist to {name}: {abs(dist):.2f} pixels")
                
if __name__ == "__main__":
    main()
