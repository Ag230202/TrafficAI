"""
pipeline.py
-----------
Integrates preprocessing → detection → tracking → lane mapping
into a single unified pipeline.

Yields structured per-frame output without holding frames in memory.
"""

from preprocessing import preprocess_pipeline, CONFIG as PREPROCESS_CONFIG
from detector import VehicleDetector, DETECTOR_CONFIG
from tracker import CentroidTracker, TRACKER_CONFIG
from lane_mapper import LaneMapper, LANE_CONFIG
import numpy as np


def build_frame_output(
    frame_index: int,
    frame,
    active_tracks: list,
    lane_mapper: LaneMapper,
) -> dict:
    """
    Assembles the final structured output dict for one frame.

    Converts Track objects into serialisable vehicle dicts,
    assigns lanes, counts per lane, and detects emergency lanes.

    Returns:
    {
        "frame_id":      int,
        "lane_counts":   {"lane_1": int, ...},
        "vehicles":      [ { "id", "lane", "bbox", "class",
                             "confidence", "direction" }, ... ],
        "emergency_lane": str | None,
        "debug_frame":   np.ndarray
    }
    """
    import cv2

    vehicles = []

    for track in active_tracks:
        # Assign this vehicle to a lane based on its current bbox center
        lane = lane_mapper.assign_lane(track.bbox)

        vehicles.append({
            "id":          track.track_id,
            "lane":        lane,
            "bbox":        track.bbox,
            "class":       track.cls,
            "confidence":  track.conf,
            "direction":   track.direction,
            # Include centroid history for heuristic emergency detection
            "centroid":      track.centroid,
            "prev_centroid": track.prev_centroid,
        })

    lane_counts, emergency_lane = lane_mapper.analyse(vehicles)

    # Create debug frame copy for drawing
    debug_frame = frame.copy()

  
    # Draw lane polygons
    for lane_name, polygon in lane_mapper.get_lane_boundaries().items():
        pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(debug_frame, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

        label_x, label_y = polygon[0]
        cv2.putText(
            debug_frame,
            lane_name,
            (label_x + 5, label_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )  # Draw vehicle boxes
        for v in vehicles:
            x1, y1, x2, y2 = v["bbox"]
            label = f'{v["id"]} {v["class"]} {v["lane"]}'

            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                debug_frame,
                label,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        # Strip internal centroid keys before returning final output
        for v in vehicles:
            v.pop("centroid", None)
            v.pop("prev_centroid", None)

        return {
            "frame_id":       frame_index,
            "lane_counts":    lane_counts,
            "vehicles":       vehicles,
            "emergency_lane": emergency_lane,
            "debug_frame":    debug_frame,
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

    Args:
        frames_folder:      Path to folder of extracted image frames
        preprocess_config:  Override for preprocessing settings
        detector_config:    Override for YOLO detector settings
        tracker_config:     Override for centroid tracker settings
        lane_config:        Override for lane boundary definitions

    Yields:
        dict — per-frame output (see build_frame_output docstring)
    """
    # ── Initialise modules once ──────────────────────────────────
    print("[Pipeline] Initialising modules...")

    detector   = VehicleDetector(detector_config or DETECTOR_CONFIG)
    tracker    = CentroidTracker(tracker_config  or TRACKER_CONFIG)
    lane_mapper = LaneMapper(lane_config         or LANE_CONFIG)
    #print("[Pipeline] Lane config:", lane_mapper.get_lane_boundaries())
    print("[Pipeline] All modules ready. Starting frame processing...\n")

    # ── Process frames via generator ────────────────────────────
    for frame_index, frame in preprocess_pipeline(
        folder_path=frames_folder,
        config=preprocess_config or PREPROCESS_CONFIG,
    ):
        # 1. Detect vehicles in this frame
        raw_detections = detector.detect(frame, frame_index)
        print(f"[Pipeline] Frame {frame_index}: detections={len(raw_detections)}")
        # 2. Update tracker with new detections → get stable tracks

        active_tracks = tracker.update(raw_detections)
        print(f"[Pipeline] Frame {frame_index}: tracks={len(active_tracks)}")

        # 3. Assemble structured output
        frame_output = build_frame_output(frame_index,frame, active_tracks, lane_mapper)
        yield frame_output   # Caller consumes result; frame is discarded
