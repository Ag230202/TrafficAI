"""
pipeline.py
-----------
Integrates preprocessing → detection → tracking → lane mapping
into a single unified pipeline.

Yields structured per-frame output without holding frames in memory.

Update: now includes EmergencyLightDetector alongside YOLO-based
emergency detection — catches fire trucks and ambulances by their
flashing lights even when YOLO misses them due to night/glare.

Update: now includes CollisionDetector — checks every vehicle pair per
frame for bounding box overlap + closing velocity and logs accidents.
"""

from preprocessing import preprocess_pipeline, CONFIG as PREPROCESS_CONFIG
from detector import VehicleDetector, DETECTOR_CONFIG
from tracker import CentroidTracker, TRACKER_CONFIG
from lane_mapper import LaneMapper, LANE_CONFIG
from emergency_detector import EmergencyLightDetector
from collision_detector import CollisionDetector
import numpy as np


def build_frame_output(
    frame_index: int,
    frame_bgr,
    frame_rgb,
    active_tracks: list,
    lane_mapper: LaneMapper,
    emergency_light_detector: EmergencyLightDetector,
    collision_detector: CollisionDetector,
    crash_detector=None
) -> dict:
    import cv2
    vehicles = []

    # Initialize or fetch track history to compute centroids across frames
    if not hasattr(build_frame_output, "_track_history"):
        build_frame_output._track_history = {}
    history = build_frame_output._track_history

    for track in active_tracks:
        # ByteTrack returns dictionaries
        bbox = track.get("bbox")
        lane = lane_mapper.assign_lane(bbox)
        track_id = track.get("id")

        # Compute current centroid
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        centroid = (cx, cy)

        # Get previous centroid and update history
        prev_centroid = history.get(track_id)
        history[track_id] = centroid

        vehicles.append({
            "id":            track_id,
            "lane":          lane,
            "bbox":          bbox,
            "centroid":      centroid,
            "prev_centroid": prev_centroid,
            "class":         track.get("class"),
            "confidence":    track.get("confidence"),
            "direction":     "stable", # ByteTrack handles stability
        })

    # ── YOLO-based emergency detection (size + speed heuristic) ─
    lane_counts, yolo_emergency_lanes, yolo_emergency_ids = lane_mapper.analyse(vehicles)

    # ── Create debug frame (RGB copy for drawing) ────────────────
    debug_frame = frame_rgb.copy()

    # ── Light-based emergency detection ─────────────────────────
    # Runs on the BGR frame for accurate HSV colour analysis.
    # Draws orange boxes on debug_frame directly.
    emergency_output = emergency_light_detector.detect(
        frame_bgr, lane_mapper, debug_frame, vehicles
    )
    detected_blobs = emergency_output["detected_blobs"]
    matched_vehicle_ids = emergency_output["matched_vehicle_ids"]
    light_emergency_lanes = list(set(b["lane"] for b in detected_blobs))

    # ── Merge both emergency sources ─────────────────────────────
    # Maintain a persistent set of known emergency IDs
    if not hasattr(build_frame_output, "_known_emergency_ids"):
        build_frame_output._known_emergency_ids = set()
    known_emergencies = build_frame_output._known_emergency_ids
    
    # Merge newly detected emergency vehicle IDs
    new_emergency_ids = set(yolo_emergency_ids).union(matched_vehicle_ids)
    known_emergencies.update(new_emergency_ids)
    
    # Either method flagging a lane is enough to trigger alert.
    all_emergency_lanes = list(set(yolo_emergency_lanes + light_emergency_lanes))

    # Re-inject lanes for any currently tracked vehicle that was PREVIOUSLY identified as an emergency vehicle
    for v in vehicles:
        if v.get("id") in known_emergencies and v.get("lane"):
            all_emergency_lanes.append(v["lane"])
            
    # Deduplicate
    all_emergency_lanes = list(set(all_emergency_lanes))
    all_emergency_vehicle_ids = known_emergencies.intersection(set(v.get("id") for v in vehicles))

    # ── Collision detection ──────────────────────────────────────
    # Run BEFORE stripping centroid keys — collision detector needs them.
    # Draws red overlap rectangles on debug_frame directly.
    collisions = collision_detector.detect(vehicles, frame_index, debug_frame)

    colors = {
        "left_road": (0, 0, 255),    # Blue in RGB
        "bottom_road": (0, 255, 0),  # Green in RGB
        "right_road": (255, 0, 0),   # Red in RGB
        "top_road": (255, 255, 0),   # Yellow in RGB
        "intersection_center": (255, 0, 255) # Magenta in RGB
    }
    
    for lane_name, polygon in lane_mapper.get_lane_boundaries().items():
        pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
        color = colors.get(lane_name, (0, 255, 255))
        
        # draw filled transparent polygon
        overlay = debug_frame.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.3, debug_frame, 0.7, 0, debug_frame)
        
        # draw border
        cv2.polylines(debug_frame, [pts], isClosed=True, color=color, thickness=2)

        label_x, label_y = polygon[0]
        cv2.putText(
            debug_frame, lane_name,
            (label_x + 5, label_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )

    # ── Draw vehicle bounding boxes ──────────────────────────────
    for v in vehicles:
        x1, y1, x2, y2 = v["bbox"]
        label = f'{v["id"]} {v["class"]} {v["lane"]}'
        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            debug_frame, label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )

    # ── Draw emergency lane overlay ──────────────────────────────
    for lane_name, polygon in lane_mapper.get_lane_boundaries().items():
        if lane_name in all_emergency_lanes:
            pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
            overlay = debug_frame.copy()
            cv2.fillPoly(overlay, [pts], (255, 80, 0))
            cv2.addWeighted(overlay, 0.25, debug_frame, 0.75, 0, debug_frame)
            cv2.polylines(debug_frame, [pts], isClosed=True, color=(255, 80, 0), thickness=3)

    # ── Strip internal centroid keys ─────────────────────────────
    for v in vehicles:
        v.pop("centroid", None)
        v.pop("prev_centroid", None)

    return {
        "frame_id":            frame_index,
        "lane_counts":         lane_counts,
        "vehicles":            vehicles,
        "emergency_lane":      all_emergency_lanes,
        "emergency_veh_ids":   all_emergency_vehicle_ids,
        "collisions":          collisions,
        "debug_frame":         debug_frame,
    }


def run_pipeline(
    frames_folder: str,
    preprocess_config: dict = None,
    detector_config: dict   = None,
    tracker_config: dict    = None,
    lane_config: dict       = None,
):
    """
    Main unified pipeline generator.

    Initialises all modules once, then processes each frame in sequence.
    Yields one structured output dict per processed frame.
    """
    print("[Pipeline] Initialising modules...")
    build_frame_output._track_history = {}

    detector            = VehicleDetector(detector_config or DETECTOR_CONFIG)
    lane_mapper         = LaneMapper(lane_config          or LANE_CONFIG)
    emergency_light_det = EmergencyLightDetector()
    collision_det       = CollisionDetector()

    print("[Pipeline] All modules ready. Starting frame processing...\n")
    
    # --- Cumulative Stats State ---
    stats = {
        "total_vehicles": 0,
        "lane_totals": {},
        "seen_vehicle_ids": set(),
        "seen_entries": set()
    }

    import cv2
    import os
    cfg = preprocess_config or PREPROCESS_CONFIG
    frame_skip   = cfg.get("frame_skip", 3)
    resize_w     = cfg.get("resize_width", 1280)
    resize_h     = cfg.get("resize_height", 720)
    valid_ext    = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")

    filenames = sorted(
        f for f in os.listdir(frames_folder)
        if f.lower().endswith(valid_ext)
    )

    from preprocessing import preprocess_for_yolo

    use_clahe = cfg.get("use_clahe", None)

    for frame_index, filename in enumerate(filenames):
        if frame_index % frame_skip != 0:
            continue

        filepath = os.path.join(frames_folder, filename)
        frame_bgr = cv2.imread(filepath)
        if frame_bgr is None:
            continue

        frame_bgr = cv2.resize(frame_bgr, (resize_w, resize_h))
        frame_rgb = preprocess_for_yolo(frame_bgr, force_clahe=use_clahe)

        # 1. Detect & Track with ByteTrack
        active_tracks = detector.detect(frame_rgb, frame_index)

        # 2. Assemble output (both BGR and RGB passed)
        frame_output = build_frame_output(
            frame_index, frame_bgr, frame_rgb,
            active_tracks, lane_mapper,
            emergency_light_det, collision_det,
        )

        # 3. Update Cumulative Stats (Line-Crossing Style)
        for v in frame_output["vehicles"]:
            vid = v.get("id")
            lane = v.get("lane")
            if vid is not None and vid != -1 and lane:
                if vid not in stats["seen_vehicle_ids"]:
                    stats["seen_vehicle_ids"].add(vid)
                    stats["total_vehicles"] += 1
                
                # Flow entry tracking
                lane_key = f"{vid}_{lane}"
                if lane_key not in stats["seen_entries"]:
                    stats["seen_entries"].add(lane_key)
                    stats["lane_totals"][lane] = stats["lane_totals"].get(lane, 0) + 1
        
        frame_output["cumulative_stats"] = stats.copy()
        yield frame_output
