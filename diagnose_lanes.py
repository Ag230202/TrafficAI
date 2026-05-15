"""
diagnose_lanes.py — v2
Run on a REAL high-quality PNG frame to see actual detection performance.
"""
import cv2
import numpy as np
from dotenv import load_dotenv
load_dotenv()

from detector import VehicleDetector, DETECTOR_CONFIG
from lane_mapper import LaneMapper, MIN_LANE_VEHICLE_AREA
from preprocessing import apply_clahe, reduce_noise, convert_bgr_to_rgb

# ── Use REAL high-quality frame, not the GIF extract ────────────
FRAME_PATH = r"D:\Traffic_AI\Traffic_Footage_Sanity\frame_000010.png"
RESIZE_W, RESIZE_H = 1280, 720

# ── UPDATED polygons (matching lane_mapper.py) ──────────────────
LANE_CONFIG = {
    "left_road": [(8,399),(397,256),(492,720),(0,720)],
    "bottom_road": [(1083,411),(1269,635),(1277,714),(801,717)],
    "right_road": [(1005,81),(1100,134),(1100,400),(1028,304),(784,164)],
    "top_road": [(405,51),(466,40),(723,159),(455,209)]
}

def main():
    frame_bgr = cv2.imread(FRAME_PATH)
    if frame_bgr is None:
        print(f"[ERROR] Cannot load: {FRAME_PATH}")
        return
    frame_bgr = cv2.resize(frame_bgr, (RESIZE_W, RESIZE_H), interpolation=cv2.INTER_LINEAR)

    # Preprocess for YOLO
    frame_rgb = frame_bgr.copy()
    frame_rgb = apply_clahe(frame_rgb, clip_limit=2.0, tile_grid=(8, 8))
    frame_rgb = reduce_noise(frame_rgb, (3, 3))
    frame_rgb = convert_bgr_to_rgb(frame_rgb)

    # Detect with lowered threshold to see what YOLO picks up
    detector_cfg = {**DETECTOR_CONFIG, "confidence_threshold": 0.15, "imgsz": 1280}
    detector = VehicleDetector(detector_cfg)
    detections = detector.detect(frame_rgb, 0)

    print(f"\n{'='*70}")
    print(f"  DETECTION DIAGNOSTICS — {len(detections)} raw detections")
    print(f"  Frame: {FRAME_PATH}")
    print(f"{'='*70}\n")

    lane_mapper = LaneMapper(LANE_CONFIG)

    lane_hits = {lane: [] for lane in LANE_CONFIG}
    lane_hits["None (unassigned)"] = []

    for i, det in enumerate(detections):
        bbox = det["bbox"]
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        area = (x2 - x1) * (y2 - y1)

        inside_any = False
        for lane_name, polygon in LANE_CONFIG.items():
            poly_np = np.array(polygon, dtype=np.int32)
            dist = cv2.pointPolygonTest(poly_np, (cx, cy), True)
            if dist >= 0:
                inside_any = True
                break

        assigned = lane_mapper.assign_lane(bbox)
        area_ok = area >= MIN_LANE_VEHICLE_AREA

        status_parts = []
        if not inside_any:
            status_parts.append("OUTSIDE polygons")
        if not area_ok:
            status_parts.append(f"AREA<{MIN_LANE_VEHICLE_AREA}")
        if assigned is None:
            status_parts.append("UNASSIGNED")
        status = " | ".join(status_parts) if status_parts else "OK"

        print(f"  [{i:>3}] {det['class']:<8} conf={det['confidence']:.2f}  "
              f"bbox=[{x1:>4},{y1:>4},{x2:>4},{y2:>4}]  "
              f"cx,cy=({cx:>4},{cy:>4})  area={area:>6}  "
              f"lane={str(assigned):<14}  {status}")

        key = assigned if assigned else "None (unassigned)"
        lane_hits[key].append(det)

    print(f"\n{'='*70}")
    print(f"  LANE SUMMARY (updated polygons)")
    print(f"{'='*70}")
    for lane, dets in lane_hits.items():
        ct = len(dets)
        ok = sum(1 for d in dets if (d['bbox'][2]-d['bbox'][0])*(d['bbox'][3]-d['bbox'][1]) >= MIN_LANE_VEHICLE_AREA)
        print(f"  {lane:<22}  total={ct:<3}  (area_ok={ok}, filtered={ct-ok})")

    # Polygon coverage
    print(f"\n{'='*70}")
    print(f"  POLYGON COVERAGE")
    print(f"{'='*70}")
    for name, pts in LANE_CONFIG.items():
        pts_np = np.array(pts, dtype=np.int32)
        area = cv2.contourArea(pts_np)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        print(f"  {name:<14}  area={area:>8.0f}px²  "
              f"x=[{min(xs):>4}-{max(xs):>4}]  y=[{min(ys):>4}-{max(ys):>4}]  "
              f"coverage={area/(RESIZE_W*RESIZE_H)*100:.1f}%")

    # Draw diagnostic image
    output = frame_bgr.copy()
    colors = {
        "left_road": (255,0,0), "bottom_road": (0,255,0),
        "right_road": (0,0,255), "top_road": (0,255,255)
    }
    for name, pts in LANE_CONFIG.items():
        pts_np = np.array(pts, np.int32)
        overlay = output.copy()
        cv2.fillPoly(overlay, [pts_np], colors[name])
        output = cv2.addWeighted(overlay, 0.3, output, 0.7, 0)
        cv2.polylines(output, [pts_np], True, colors[name], 2)
        lx, ly = pts[0]
        cv2.putText(output, name, (lx+5, ly+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[name], 2)

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cx, cy = (x1+x2)//2, (y1+y2)//2
        assigned = lane_mapper.assign_lane(det["bbox"])
        color = (0, 255, 0) if assigned else (0, 0, 255)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.circle(output, (cx, cy), 4, color, -1)
        label = f"{det['class']} -> {assigned or 'NONE'}"
        cv2.putText(output, label, (x1, max(15, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    out_path = r"D:\Traffic_AI\diagnostic_output_v2.jpg"
    cv2.imwrite(out_path, output)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
