"""
collision_detector.py
---------------------
Detects vehicle collisions by analysing bounding box overlap and
closing velocity between tracked vehicle pairs across frames.

Strategy:
  A collision is flagged when two tracked vehicles satisfy ALL of:
    1. Their bounding boxes overlap (IoU >= iou_threshold), AND
    2. They were moving TOWARD each other in the previous frame
       (closing velocity >= min_closing_speed pixels/frame), AND
    3. Both tracks have motion history (prev_centroid is not None) so
       the velocity check is meaningful.

  Condition 2 filters out vehicles that are merely stationary beside
  each other or moving in the same direction — only converging pairs
  with actual overlap are flagged.

  A cooldown (collision_cooldown_frames) prevents the same pair from
  being logged repeatedly across consecutive frames for the same event.

Integration:
  - Instantiate CollisionDetector once in pipeline.py alongside the
    other detectors.
  - Call detect(vehicles, frame_id, debug_frame) once per frame inside
    build_frame_output().
  - The returned collision list is added to the frame_output dict under
    the key "collisions".
  - CollisionLogger (used in example_usage.py) writes every collision
    to a timestamped log file and accumulates summary stats.
"""

import cv2
import numpy as np
import logging
import os
from datetime import datetime


# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
COLLISION_CONFIG = {
    # Minimum bounding box overlap (Intersection over Union) to consider
    # two vehicles as physically occupying the same space.
    # 0.0 = any overlap at all; 0.1 = 10% overlap required.
    # Keep low (0.05–0.15) — at 640x480 even a real collision shows modest IoU.
    "iou_threshold": 0.15,

    # Minimum closing speed in pixels/frame for both vehicles combined.
    # Filters out stationary neighbours and slow lane-merges.
    # At frame_skip=3 and ~30fps, 1 frame ≈ 100ms, so 8px/frame ≈ ~2.9km/h
    # at typical CCTV scale. Tune upward if parked cars trigger false positives.
    "min_closing_speed": 15.0,

    # Frames to suppress re-flagging the same vehicle pair after a collision.
    # At frame_skip=3 this is roughly: cooldown * 3 / 30 seconds.
    # Default 10 → suppresses for ~1 second of real video.
    "collision_cooldown_frames": 10,

    # Draw collision markers on the debug frame.
    "draw_debug": True,

    # Log file path. Set to None to disable file logging.
    "log_file": "collision_log.txt",
}


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def _compute_iou(bbox_a: list, bbox_b: list) -> float:
    """
    Computes Intersection over Union between two bounding boxes.
    bbox format: [x1, y1, x2, y2]
    Returns float in [0.0, 1.0].
    """
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    inter   = inter_w * inter_h

    if inter == 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union  = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def _closing_speed(vehicle_a: dict, vehicle_b: dict) -> float:
    """
    Returns the combined closing speed of two vehicles in pixels/frame.

    Closing speed = rate at which the distance between centroids is
    decreasing. Positive = moving toward each other. Negative = moving
    apart. Uses prev_centroid and centroid from each vehicle dict.

    Returns 0.0 if either vehicle has no motion history.
    """
    curr_a = vehicle_a.get("centroid")
    prev_a = vehicle_a.get("prev_centroid")
    curr_b = vehicle_b.get("centroid")
    prev_b = vehicle_b.get("prev_centroid")

    if None in (curr_a, prev_a, curr_b, prev_b):
        return 0.0

    def dist(p, q):
        return np.hypot(p[0] - q[0], p[1] - q[1])

    prev_gap = dist(prev_a, prev_b)
    curr_gap = dist(curr_a, curr_b)

    # Positive = gap shrinking = closing toward each other
    return prev_gap - curr_gap


# ─────────────────────────────────────────────
#  COLLISION DETECTOR
# ─────────────────────────────────────────────
class CollisionDetector:
    """
    Detects collisions between tracked vehicle pairs.

    Usage:
        detector = CollisionDetector()
        collisions = detector.detect(vehicles, frame_id, debug_frame)

    Each element of the returned list is a dict:
        {
            "frame_id":      int,
            "vehicle_a_id":  int,
            "vehicle_b_id":  int,
            "vehicle_a_cls": str,
            "vehicle_b_cls": str,
            "iou":           float,
            "closing_speed": float,
            "lane":          str or None,
            "bbox_overlap":  [x1, y1, x2, y2],   # intersection rectangle
        }
    """

    def __init__(self, config: dict = None):
        self.cfg      = config or COLLISION_CONFIG
        # Maps frozenset({id_a, id_b}) → last frame_id the pair was flagged.
        # Used to enforce the cooldown window.
        self._cooldown: dict = {}

    def detect(
        self,
        vehicles: list,
        frame_id: int,
        debug_frame=None,
    ) -> list:
        """
        Checks all vehicle pairs in the current frame for collision.

        vehicles:    list of vehicle dicts from build_frame_output()
                     Each dict must have: id, bbox, centroid,
                     prev_centroid, class, lane.
        frame_id:    current frame index (used for cooldown tracking).
        debug_frame: optional RGB np.ndarray to draw markers on.

        Returns list of collision dicts (may be empty).
        """
        cfg        = self.cfg
        iou_thresh = cfg["iou_threshold"]
        min_speed  = cfg["min_closing_speed"]
        cooldown   = cfg["collision_cooldown_frames"]
        collisions = []

        # Need at least 2 vehicles to have a collision
        if len(vehicles) < 2:
            return collisions

        # Check every unique pair (i, j) where i < j
        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):
                v_a = vehicles[i]
                v_b = vehicles[j]

                id_a = v_a.get("id")
                id_b = v_b.get("id")
                pair_key = frozenset({id_a, id_b})

                # ── Cooldown check ───────────────────────────────
                last_flagged = self._cooldown.get(pair_key)
                if last_flagged is not None:
                    if frame_id - last_flagged < cooldown:
                        continue  # still within suppression window

                # ── Condition 1: bounding box overlap ───────────
                iou = _compute_iou(v_a["bbox"], v_b["bbox"])
                if iou < iou_thresh:
                    continue

                # ── Condition 2: vehicles were closing toward each other ──
                speed = _closing_speed(v_a, v_b)
                if speed < min_speed:
                    continue

                # ── Collision confirmed ──────────────────────────
                self._cooldown[pair_key] = frame_id

                # Compute overlap rectangle for debug drawing
                ax1, ay1, ax2, ay2 = v_a["bbox"]
                bx1, by1, bx2, by2 = v_b["bbox"]
                overlap_box = [
                    max(ax1, bx1), max(ay1, by1),
                    min(ax2, bx2), min(ay2, by2),
                ]

                collision = {
                    "frame_id":      frame_id,
                    "vehicle_a_id":  id_a,
                    "vehicle_b_id":  id_b,
                    "vehicle_a_cls": v_a.get("class", "unknown"),
                    "vehicle_b_cls": v_b.get("class", "unknown"),
                    "iou":           round(iou, 4),
                    "closing_speed": round(speed, 2),
                    "lane":          v_a.get("lane"),   # lane of first vehicle
                    "bbox_overlap":  overlap_box,
                }
                collisions.append(collision)

                # ── Draw on debug frame ──────────────────────────
                if debug_frame is not None and cfg.get("draw_debug", True):
                    ox1, oy1, ox2, oy2 = overlap_box
                    # Red filled overlap rectangle (semi-transparent)
                    overlay = debug_frame.copy()
                    cv2.rectangle(overlay, (ox1, oy1), (ox2, oy2), (255, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.4, debug_frame, 0.6, 0, debug_frame)
                    # Red border around overlap
                    cv2.rectangle(debug_frame, (ox1, oy1), (ox2, oy2), (255, 0, 0), 2)
                    # Label
                    label = f"COLLISION {id_a}&{id_b}"
                    cv2.putText(
                        debug_frame, label,
                        (ox1, max(15, oy1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 0), 2,
                    )

        return collisions

    def reset_cooldowns(self):
        """Clears the cooldown table. Call between unrelated video sequences."""
        self._cooldown.clear()


# ─────────────────────────────────────────────
#  COLLISION LOGGER
# ─────────────────────────────────────────────
class CollisionLogger:
    """
    Logs collision events to console and an optional text file,
    and accumulates summary statistics for the pipeline summary.

    Usage in example_usage.py:
        logger = CollisionLogger(log_file="collision_log.txt")
        ...
        # inside the frame loop:
        logger.log(frame_output.get("collisions", []))
        ...
        # after the loop:
        logger.print_summary()
    """

    def __init__(self, log_file: str = None):
        self.log_file          = log_file
        self.total_collisions  = 0
        self.collision_frames  = []        # frame_ids where collision detected
        self.collision_events  = []        # full list of collision dicts
        self.lane_counts       = {}        # lane → collision count
        self.pair_counts       = {}        # "id_a&id_b" → count

        # Set up file logger
        self._file_logger = None
        if log_file:
            self._setup_file_logger(log_file)

    def _setup_file_logger(self, path: str):
        logger = logging.getLogger("CollisionLogger")
        logger.setLevel(logging.INFO)
        # Avoid duplicate handlers if instantiated multiple times
        if not logger.handlers:
            fh = logging.FileHandler(path, mode="a", encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s",
                                               datefmt="%Y-%m-%d %H:%M:%S"))
            logger.addHandler(fh)
        self._file_logger = logger
        self._file_logger.info("=" * 60)
        self._file_logger.info("  COLLISION LOG — session started")
        self._file_logger.info("=" * 60)

    def log(self, collisions: list):
        """
        Call once per frame with the list returned by CollisionDetector.detect().
        Logs each collision to console and file, updates statistics.
        """
        for c in collisions:
            self.total_collisions += 1
            fid = c["frame_id"]

            if fid not in self.collision_frames:
                self.collision_frames.append(fid)

            self.collision_events.append(c)

            # Lane tally
            lane = c.get("lane") or "unknown"
            self.lane_counts[lane] = self.lane_counts.get(lane, 0) + 1

            # Pair tally
            pair_str = f"{c['vehicle_a_id']}&{c['vehicle_b_id']}"
            self.pair_counts[pair_str] = self.pair_counts.get(pair_str, 0) + 1

            # Console output
            msg = (
                f"[ACCIDENT] Frame {fid:>5} | "
                f"Vehicles {c['vehicle_a_id']}({c['vehicle_a_cls']}) "
                f"& {c['vehicle_b_id']}({c['vehicle_b_cls']}) | "
                f"Lane: {lane} | "
                f"IoU: {c['iou']:.3f} | "
                f"Closing speed: {c['closing_speed']:.1f}px/frame"
            )
            print(msg)

            # File output
            if self._file_logger:
                self._file_logger.info(msg)

    def print_summary(self):
        """Prints a collision summary block in the same style as PipelineSummary."""
        print()
        print("─" * 55)
        print("  COLLISION SUMMARY")
        print("─" * 55)
        print(f"  Total collisions        : {self.total_collisions}")
        print(f"  Frames with collision   : {len(self.collision_frames)}")

        if self.collision_frames:
            print(f"  Frame IDs               : {self.collision_frames}")

        if self.lane_counts:
            print()
            print("  Collisions by lane:")
            for lane, count in sorted(self.lane_counts.items(),
                                      key=lambda x: -x[1]):
                bar = "█" * min(count, 40)
                print(f"    {lane:<12} {count:>4}  {bar}")

        if self.pair_counts:
            print()
            print("  Most frequent pairs:")
            for pair, count in sorted(self.pair_counts.items(),
                                      key=lambda x: -x[1])[:5]:
                print(f"    Vehicles {pair:<10} {count} event(s)")

        if self.log_file and self.total_collisions > 0:
            print()
            print(f"  Log file                : {self.log_file}")

        print("─" * 55)

        if self._file_logger:
            self._file_logger.info("─" * 60)
            self._file_logger.info(f"  Session ended — {self.total_collisions} collision(s) detected")
            self._file_logger.info("─" * 60)
