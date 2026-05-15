"""
diagnose_lanes.py — v3
Run on frame_000001.png with overlap detection to find overcounting causes.
"""
import cv2
import numpy as np
from dotenv import load_dotenv
load_dotenv()

from detector import VehicleDetector, DETECTOR_CONFIG
from lane_mapper import LaneMapper, MIN_LANE_VEHICLE_AREA
from preprocessing import apply_clahe, reduce_noise, convert_bgr_to_rgb

FRAME_PATH = r"D:\Traffic_AI\aa\frame_000001.png"
RESIZE_W, RESIZE_H = 1280, 720

LANE_CONFIG = {
    "left_road": [(437, 222), (621, 698), (0, 720), (0, 416)],
    "top_road": [(455, 198), (685, 144), (509, 53), (414, 67)],
    "right_road": [(782, 156), (1033, 317), (1078, 139), (1009, 81)],
    "bottom_road": [(1056, 359), (650, 708), (1209, 706), (1277, 636)]
}

def calculate_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    xi1, yi1 = max(x1_1, x1_2), max(y1_1, y1_2)
    xi2, yi2 = min(x2_1, x2_2), min(y2_1, y2_2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    return inter_area / float(box1_area + box2_area - inter_area)

def main():
    frame_bgr = cv2.imread(FRAME_PATH)
    if frame_bgr is None:
        print(f"[ERROR] Cannot load: {FRAME_PATH}")
        return
    frame_bgr = cv2.resize(frame_bgr, (RESIZE_W, RESIZE_H), interpolation=cv2.INTER_LINEAR)

    frame_rgb = frame_bgr.copy()
    frame_rgb = apply_clahe(frame_rgb, clip_limit=2.0, tile_grid=(8, 8))
    frame_rgb = reduce_noise(frame_rgb, (3, 3))
    frame_rgb = convert_bgr_to_rgb(frame_rgb)

    detector_cfg = {**DETECTOR_CONFIG, "confidence_threshold": 0.15, "imgsz": 1280}
    detector = VehicleDetector(detector_cfg)
    detections = detector.detect(frame_rgb, 1)

    print(f"\n{'='*70}")
    print(f"  DETECTION DIAGNOSTICS — {len(detections)} raw detections")
    print(f"  Frame: {FRAME_PATH}")
    print(f"{'='*70}\n")

    lane_mapper = LaneMapper(LANE_CONFIG)
    lane_hits = {lane: [] for lane in LANE_CONFIG}
    lane_hits["None (unassigned)"] = []

    # Check for overlaps
    overlaps = []
    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):
            iou = calculate_iou(detections[i]["bbox"], detections[j]["bbox"])
            if iou > 0.5:
                overlaps.append((i, j, iou))

    for i, det in enumerate(detections):
        bbox = det["bbox"]
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        area = (x2 - x1) * (y2 - y1)
        
        assigned = lane_mapper.assign_lane(bbox)
        area_ok = area >= MIN_LANE_VEHICLE_AREA
        
        overlap_info = ""
        for o1, o2, iou in overlaps:
            if i == o1: overlap_info = f"[OVERLAP with {o2} (IoU={iou:.2f})]"
            if i == o2: overlap_info = f"[OVERLAP with {o1} (IoU={iou:.2f})]"

        print(f"  [{i:>3}] {det['class']:<8} conf={det['confidence']:.2f}  "
              f"centroid=({cx:>4},{cy:>4})  area={area:>6}  "
              f"lane={str(assigned):<14}  {overlap_info}")

        key = assigned if assigned else "None (unassigned)"
        lane_hits[key].append(det)

    print(f"\n{'='*70}")
    print(f"  LANE SUMMARY")
    print(f"{'='*70}")
    for lane, dets in lane_hits.items():
        print(f"  {lane:<22}  total={len(dets)}")

    output = frame_bgr.copy()
    for name, pts in LANE_CONFIG.items():
        pts_np = np.array(pts, np.int32)
        cv2.polylines(output, [pts_np], True, (0,255,255), 2)
        cv2.putText(output, name, (pts[0][0], pts[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]
        assigned = lane_mapper.assign_lane(det["bbox"])
        color = (0, 255, 0) if assigned else (0, 0, 255)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        label = f"#{i} {det['class']} -> {assigned or 'NONE'}"
        cv2.putText(output, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    out_path = r"D:\Traffic_AI\diagnostic_output_v3.jpg"
    cv2.imwrite(out_path, output)
    print(f"\n  Diagnostic saved: {out_path}")

if __name__ == "__main__":
    main()
