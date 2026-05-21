import cv2 as cv
import numpy as np
import math
from ultralytics import YOLO
import argparse
import os

# Configuration defaults (can be overridden via command line)
DEFAULT_IOU_THRESHOLD = 0.40 # Increased significantly to ignore perspective overlaps
DEFAULT_MIN_SPEED = 6.0      # Increased to ignore bounding box jitter
DEFAULT_PERSIST_FRAMES = 3   # Number of consecutive frames to confirm crash

# Load default path from .env if present
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
DEFAULT_PATH = "cctv_crash.mp4"
if os.path.exists(env_path):
    with open(env_path, "r") as f:
        for line in f:
            if line.startswith("FRAMES_FOLDER_PATH="):
                DEFAULT_PATH = line.strip().split("=", 1)[1]
                break


def compute_iou(box_a, box_b):
    """Compute Intersection over Union between two boxes.
    Boxes are (x1, y1, x2, y2)."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def centroid(box):
    """Return centroid (cx, cy) of a bounding box."""
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    return np.array([cx, cy])


def main(args):
    # Load YOLO model
    model = YOLO(args.model_path)
    # SORT is replaced by YOLO's built-in tracking

    is_dir = os.path.isdir(args.video_path)
    if is_dir:
        image_files = sorted([
            os.path.join(args.video_path, f) 
            for f in os.listdir(args.video_path) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        print(f"Reading {len(image_files)} frames from directory: {args.video_path}")
    else:
        cap = cv.VideoCapture(args.video_path)
        if not cap.isOpened():
            print(f"Cannot open video: {args.video_path}")
            return
        print(f"Reading video file: {args.video_path}")

    # Keep previous positions per track ID to compute speed
    prev_positions = {}
    # Crash persistence counter per pair of IDs
    crash_counters = {}

    frame_idx = 0
    while True:
        if is_dir:
            if frame_idx >= len(image_files):
                break
            frame = cv.imread(image_files[frame_idx])
            if frame is None:
                frame_idx += 1
                continue
        else:
            ret, frame = cap.read()
            if not ret:
                break
        
        frame_idx += 1
        frame = cv.resize(frame, (1280, 720))

        # Run detection and tracking
        results = model.track(
            frame, 
            persist=True, 
            tracker="bytetrack.yaml", 
            conf=args.confidence,
            stream=False,
            verbose=False
        )

        # Dictionary of current track boxes keyed by ID
        current_boxes = {}
        if results and results[0].boxes and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy()
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                current_boxes[int(track_id)] = (x1, y1, x2, y2)

        # Compare every pair of tracked objects
        ids = list(current_boxes.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id_a, id_b = ids[i], ids[j]
                box_a = current_boxes[id_a]
                box_b = current_boxes[id_b]
                iou = compute_iou(box_a, box_b)
                if iou < args.iou_threshold:
                    continue
                # Compute closing speed (centroid distance change)
                ca = centroid(box_a)
                cb = centroid(box_b)
                # Retrieve previous centroids
                prev_a = prev_positions.get(id_a)
                prev_b = prev_positions.get(id_b)
                if prev_a is not None and prev_b is not None:
                    speed_a = np.linalg.norm(ca - prev_a)
                    speed_b = np.linalg.norm(cb - prev_b)
                    dist_prev = np.linalg.norm(prev_a - prev_b)
                    dist_curr = np.linalg.norm(ca - cb)
                    
                    # They must be converging (getting closer)
                    is_converging = dist_curr < dist_prev
                    # At least one must be moving faster than jitter
                    is_moving = max(speed_a, speed_b) > args.min_speed
                    
                    if not (is_converging and is_moving):
                        continue
                        
                    closing_speed = dist_prev - dist_curr
                # Use a pair key for persistence
                pair_key = tuple(sorted((id_a, id_b)))
                crash_counters[pair_key] = crash_counters.get(pair_key, 0) + 1
                if crash_counters[pair_key] >= args.persist_frames:
                    print(f"[CRASH] Frame {frame_idx}: IDs {id_a}&{id_b} - IoU={iou:.3f}, Speed={closing_speed:.2f}")
                    # Reset counter to avoid repeated prints
                    crash_counters[pair_key] = 0
        # Reset counters for pairs not meeting criteria this frame
        # Remove entries for pairs that are no longer present in current detections
        current_pair_keys = [tuple(sorted((ids[i], ids[j]))) for i in range(len(ids)) for j in range(i + 1, len(ids))]
        for key in list(crash_counters.keys()):
            if key not in current_pair_keys:
                crash_counters.pop(key, None)

        # Update previous positions
        for tid, bbox in current_boxes.items():
            prev_positions[tid] = centroid(bbox)

        # Visualise output
        for tid, bbox in current_boxes.items():
            x1, y1, x2, y2 = bbox
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.putText(frame, str(tid), (x1, y1 - 5), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv.imshow('Crash Detector', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    if not is_dir:
        cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone YOLO+SORT crash detector")
    parser.add_argument("--video_path", default=DEFAULT_PATH, help=f"Path to input video file or frames directory (default: {DEFAULT_PATH})")
    parser.add_argument("--model_path", default="yolov8m.pt", help="Path to YOLO weights (.pt) (default: yolov8m.pt)")
    parser.add_argument("--iou_threshold", type=float, default=DEFAULT_IOU_THRESHOLD,
                        help=f"IoU threshold (default {DEFAULT_IOU_THRESHOLD})")
    parser.add_argument("--min_speed", type=float, default=DEFAULT_MIN_SPEED,
                        help=f"Minimum closing speed in pixels/frame (default {DEFAULT_MIN_SPEED})")
    parser.add_argument("--persist_frames", type=int, default=DEFAULT_PERSIST_FRAMES,
                        help="Number of consecutive frames required to confirm a crash")
    parser.add_argument("--confidence", type=float, default=0.3,
                        help="Detection confidence threshold for YOLO")
    parser.add_argument("--show", action="store_true", help="Show video with bounding boxes")
    args = parser.parse_args()
    main(args)
