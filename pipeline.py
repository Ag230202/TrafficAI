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
) -> dict:
    """
    Assembles the final structured output dict for one frame.

    frame_bgr: original BGR frame (before colour conversion) — used by
               EmergencyLightDetector for HSV colour analysis
    frame_rgb: RGB frame (after preprocessing) — used for debug display

    Returns:
    {
        "frame_id":      int,
        "lane_counts":   {"lane_name": int, ...},
        "vehicles":      [ { "id", "lane", "bbox", "class",
                             "confidence", "direction" }, ... ],
        "emergency_lane": list,
        "collisions":    list of collision dicts,
        "debug_frame":   np.ndarray (RGB)
    }
    """
    import cv2

    vehicles = []

    for track in active_tracks:
        lane = lane_mapper.assign_lane(track.bbox)

        vehicles.append({
            "id":            track.track_id,
            "lane":          lane,
            "bbox":          track.bbox,
            "class":         track.cls,
            "confidence":    track.conf,
            "direction":     track.direction,
            "centroid":      track.centroid,
            "prev_centroid": track.prev_centroid,
        })

    # ── YOLO-based emergency detection (size + speed heuristic) ─
    lane_counts, yolo_emergency_lanes, yolo_emergency_ids = lane_mapper.analyse(vehicles)

    # ── Create debug frame (RGB copy for drawing) ────────────────
    debug_frame = frame_rgb.copy()

    # ── Light-based emergency detection ─────────────────────────
    # Runs on the BGR frame for accurate HSV colour analysis.
    # Draws orange boxes on debug_frame directly.
    light_emergency_lanes = emergency_light_detector.detect(
        frame_bgr, lane_mapper, debug_frame
    )

    # ── Merge both emergency sources ─────────────────────────────
    # Either method flagging a lane is enough to trigger alert.
    all_emergency_lanes = list(set(yolo_emergency_lanes + light_emergency_lanes))

    # Merge emergency vehicle IDs from both sources
    # yolo_emergency_ids: specific truck/bus IDs from area+speed heuristic
    # light source doesn't give vehicle IDs (it works from pixel blobs)
    all_emergency_vehicle_ids = yolo_emergency_ids

    # ── Collision detection ──────────────────────────────────────
    # Run BEFORE stripping centroid keys — collision detector needs them.
    # Draws red overlap rectangles on debug_frame directly.
    collisions = collision_detector.detect(vehicles, frame_index, debug_frame)

    # ── Draw lane polygons ───────────────────────────────────────
    for lane_name, polygon in lane_mapper.get_lane_boundaries().items():
        pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(debug_frame, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

        label_x, label_y = polygon[0]
        cv2.putText(
            debug_frame, lane_name,
            (label_x + 5, label_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2
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

    detector            = VehicleDetector(detector_config or DETECTOR_CONFIG)
    tracker             = CentroidTracker(tracker_config  or TRACKER_CONFIG)
    lane_mapper         = LaneMapper(lane_config          or LANE_CONFIG)
    emergency_light_det = EmergencyLightDetector()
    collision_det       = CollisionDetector()

    print("[Pipeline] All modules ready. Starting frame processing...\n")

    # We need the BGR frame for emergency light detection, but
    # preprocess_pipeline yields RGB. So we load raw BGR here first,
    # then preprocess a copy for YOLO.
    import cv2
    import os

    cfg = preprocess_config or PREPROCESS_CONFIG
    frame_skip   = cfg.get("frame_skip", 3)
    resize_w     = cfg.get("resize_width", 640)
    resize_h     = cfg.get("resize_height", 480)
    valid_ext    = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")

    filenames = sorted(
        f for f in os.listdir(frames_folder)
        if f.lower().endswith(valid_ext)
    )

    from preprocessing import (
        apply_roi, adjust_brightness_contrast, apply_clahe,
        reduce_noise, convert_bgr_to_rgb
    )

    use_clahe = cfg.get("use_clahe", False)

    for frame_index, filename in enumerate(filenames):
        if frame_index % frame_skip != 0:
            continue

        filepath = os.path.join(frames_folder, filename)
        frame_bgr = cv2.imread(filepath)
        if frame_bgr is None:
            continue

        # Resize BGR frame (used by emergency light detector)
        frame_bgr = cv2.resize(frame_bgr, (resize_w, resize_h),
                                interpolation=cv2.INTER_LINEAR)

        # Build preprocessed RGB frame for YOLO
        frame_rgb = frame_bgr.copy()
        rois = cfg.get("rois", [])
        if rois:
            frame_rgb = apply_roi(frame_rgb, rois)

        if use_clahe:
            frame_rgb = apply_clahe(
                frame_rgb,
                clip_limit=cfg.get("clahe_clip_limit", 2.0),
                tile_grid=cfg.get("clahe_tile_grid", (8, 8)),
            )
        else:
            frame_rgb = adjust_brightness_contrast(
                frame_rgb,
                alpha=cfg.get("alpha", 1.2),
                beta=cfg.get("beta", 15),
            )

        frame_rgb = reduce_noise(frame_rgb, cfg.get("blur_kernel", (3, 3)))
        frame_rgb = convert_bgr_to_rgb(frame_rgb)

        # 1. Detect with YOLO
        raw_detections = detector.detect(frame_rgb, frame_index)
        # print(f"[Pipeline] Frame {frame_index}: detections={len(raw_detections)}")

        # 2. Track
        active_tracks = tracker.update(raw_detections)
        # print(f"[Pipeline] Frame {frame_index}: tracks={len(active_tracks)}")

        # 3. Assemble output (both BGR and RGB passed)
        frame_output = build_frame_output(
            frame_index, frame_bgr, frame_rgb,
            active_tracks, lane_mapper,
            emergency_light_det, collision_det,
        )

        yield frame_output
