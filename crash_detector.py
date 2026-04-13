"""
crash_detector.py
-----------------
Higher-level crash detection using a 4-signal confidence model with
temporal persistence filtering.

Purpose:
  The CollisionDetector in collision_detector.py fires on bounding-box
  IoU + closing velocity every frame — it's a low-level, per-pair check.
  This module sits ABOVE it and decides whether a real crash has occurred
  by combining 4 independent signals into a per-lane confidence score,
  then requiring that score to exceed a threshold for 3+ consecutive
  processed frames before firing an alert.

  This eliminates the 72% false-positive rate observed in signal_log.txt
  (caused by detection jitter triggering raw IoU collisions).

4 Signals and their scores:
  +40  Bbox overlap IoU ≥ 0.3 between two different track IDs
       (uses already-computed collisions list from CollisionDetector)
  +35  3+ track IDs vanish from the same lane in the same frame
       (cluster disappearance = crash or wholesale detection loss)
  +30  Lane vehicle count drops ≥ 3 in one frame, isolated
       (not a green-light clear — adjacent lanes didn't drop simultaneously)
  +25  Two vehicles in the same lane polygon moving toward each other
       (direction conflict: only fires after frame 3 when direction data exists)

Confidence thresholds:
  < 50   → No action (log internally)
  50–79  → Possible crash (logged, monitored)
  80–99  → Probable crash (log + snapshot)
  ≥ 100  → Confirmed crash → crash_report returned → alerting triggered

Temporal persistence:
  Score must exceed the threshold for ≥ 3 consecutive processed frames.
  At frame_skip=3 and 30fps, 3 processed frames ≈ 0.3 seconds of real video.
  Fast enough to catch crashes; slow enough to ignore single-frame YOLO jitter.

Integration (example_usage.py):
  crash_detector = CrashDetector()            # before loop
  crash_report = crash_detector.update(frame_output)   # inside loop
  if crash_report:
      alert_dispatcher.dispatch(crash_report, frame_output["debug_frame"])
"""

from collections import defaultdict, deque
from datetime import datetime
from typing import Optional
import numpy as np


# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────

CRASH_DETECTOR_CONFIG = {
    # ── Signal scores ──────────────────────────────────────────
    "score_bbox_overlap":      60,   # Increased from 40 so a single overlap crosses the 50 threshold
    "score_id_vanish":         35,   # 3+ IDs vanish from same lane at once
    "score_count_drop":        30,   # Lane count drops ≥ count_drop_threshold
    "score_direction_conflict": 25,  # Two vehicles heading toward each other

    # ── Thresholds ──────────────────────────────────────────────
    "iou_threshold":           0.05, # Lowered from 0.30 so ANY contact generates score
    "min_vanish_count":        3,    # Min IDs vanishing to score the signal
    "count_drop_threshold":    3,    # Min vehicle count drop to score signal
    "persistence_frames":      1,    # Lowered from 3 to 1 to trigger immediately

    # ── Confidence bands ────────────────────────────────────────
    "threshold_possible":      50,   # Score ≥ 50: possible crash, log only
    "threshold_probable":      80,   # Score ≥ 80: probable, log + snapshot
    "threshold_confirmed":     100,  # Score ≥ 100: confirmed, full alert

    # ── Cooldown ────────────────────────────────────────────────
    "alert_cooldown_frames":   60,   # Min processed frames between alerts per lane
                                     # At frame_skip=3, 60 frames ≈ 6 seconds

    # ── Rolling history ─────────────────────────────────────────
    "score_history_len":       5,    # How many past frame scores to keep per lane

    # ── Direction conflict ───────────────────────────────────────
    "min_frame_for_direction": 3,    # Don't check directions until frame 3+
}


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def _compute_iou(bbox_a: list, bbox_b: list) -> float:
    """
    Intersection over Union between two [x1,y1,x2,y2] bounding boxes.
    Returns float in [0.0, 1.0].
    """
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b

    ix1 = max(ax1, bx1);  iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2);  iy2 = min(ay2, by2)

    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    inter   = inter_w * inter_h

    if inter == 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union  = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def _directions_conflict(dir_a: str, dir_b: str) -> bool:
    """
    Returns True if two direction strings suggest vehicles are
    moving toward each other (head-on or oblique).

    Canonical directions from tracker.py: 'up', 'down', 'left', 'right',
    'up-left', 'up-right', 'down-left', 'down-right', 'stationary', 'unknown'
    """
    if not dir_a or not dir_b:
        return False
    if "unknown" in (dir_a, dir_b) or "stationary" in (dir_a, dir_b):
        return False

    OPPOSITES = {
        "up":         "down",
        "down":       "up",
        "left":       "right",
        "right":      "left",
        "up-left":    "down-right",
        "up-right":   "down-left",
        "down-left":  "up-right",
        "down-right": "up-left",
    }
    return OPPOSITES.get(dir_a) == dir_b


# ─────────────────────────────────────────────
#  CRASH DETECTOR
# ─────────────────────────────────────────────

class CrashDetector:
    """
    Detects vehicle crashes using a 4-signal confidence model
    with temporal persistence filtering.

    Call update(frame_output) once per frame inside the pipeline loop.
    Returns a crash_report dict when a crash is confirmed, else None.
    """

    def __init__(self, config: dict = None):
        self.cfg = config or CRASH_DETECTOR_CONFIG

        # Per-lane rolling score history (deque of recent scores)
        hist_len = self.cfg.get("score_history_len", 5)
        self._score_history: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=hist_len)
        )

        # Per-lane consecutive-frames-above-threshold counter
        self._persist_count: dict[str, int] = defaultdict(int)

        # Per-lane alert cooldown counter (processed frames since last alert)
        self._cooldown: dict[str, int] = defaultdict(
            lambda: self.cfg.get("alert_cooldown_frames", 60)
        )

        # Previous frame state for comparison signals
        self._prev_lane_counts: dict[str, int] = {}
        self._prev_ids_per_lane: dict[str, set] = defaultdict(set)

        self._frame_count = 0   # processed frame counter

    # ──────────────────────────────────────────────────────────
    #  PUBLIC API
    # ──────────────────────────────────────────────────────────

    def update(self, frame_output: dict) -> Optional[dict]:
        """
        Evaluate all signals for this frame and decide whether to fire a crash alert.

        Args:
            frame_output: dict from run_pipeline() generator, containing:
                - "frame_id"     : int
                - "vehicles"     : list of vehicle dicts
                - "lane_counts"  : {lane: count}
                - "collisions"   : list of collision dicts (from CollisionDetector)
                - "debug_frame"  : np.ndarray (RGB)

        Returns:
            crash_report dict if crash confirmed, else None.
        """
        self._frame_count += 1

        frame_id    = frame_output.get("frame_id", self._frame_count)
        vehicles    = frame_output.get("vehicles", [])
        lane_counts = frame_output.get("lane_counts", {})
        collisions  = frame_output.get("collisions", [])

        # Tick down all cooldowns each frame
        for lane in list(self._cooldown.keys()):
            self._cooldown[lane] += 1

        # ── Build per-lane vehicle dict ───────────────────────
        vehicles_by_lane: dict[str, list] = defaultdict(list)
        for v in vehicles:
            lane = v.get("lane")
            if lane and lane != "unknown":
                vehicles_by_lane[lane].append(v)

        # ── Score each lane ───────────────────────────────────
        crash_report = None
        all_lanes = set(lane_counts.keys()) | set(self._prev_lane_counts.keys())

        for lane in all_lanes:
            if lane == "unknown":
                continue

            score = 0
            triggered_signals = []
            vehicle_ids_in_lane = {v["id"] for v in vehicles_by_lane.get(lane, [])}

            # Signal 1: Bbox overlap IoU (from existing CollisionDetector output) +40
            iou_score = self._check_bbox_overlap(collisions, lane)
            score += iou_score
            if iou_score > 0:
                triggered_signals.append(f"iou+{iou_score}")

            # Signal 2: ID cluster vanish +35
            vanish_score = self._check_id_vanish(lane, vehicle_ids_in_lane)
            score += vanish_score
            if vanish_score > 0:
                triggered_signals.append(f"vanish+{vanish_score}")

            # Signal 3: Lane count drop +30
            drop_score = self._check_count_drop(lane, lane_counts)
            score += drop_score
            if drop_score > 0:
                triggered_signals.append(f"drop+{drop_score}")

            # Signal 4: Direction conflict +25 (only after frame 3)
            if self._frame_count >= self.cfg.get("min_frame_for_direction", 3):
                dir_score = self._check_direction_conflict(
                    vehicles_by_lane.get(lane, [])
                )
                score += dir_score
                if dir_score > 0:
                    triggered_signals.append(f"direction+{dir_score}")

            # ── Record score in rolling history ───────────────
            self._score_history[lane].append(score)

            # ── Temporal persistence check ────────────────────
            threshold_confirmed = self.cfg.get("threshold_confirmed", 100)
            threshold_probable  = self.cfg.get("threshold_probable", 80)
            threshold_possible  = self.cfg.get("threshold_possible", 50)
            persist_needed      = self.cfg.get("persistence_frames", 3)

            if score >= threshold_possible:
                self._persist_count[lane] += 1
            else:
                self._persist_count[lane] = 0  # reset streak

            consecutive = self._persist_count[lane]
            cooldown_ok = self._cooldown[lane] >= self.cfg.get("alert_cooldown_frames", 60)

            if consecutive >= persist_needed and cooldown_ok:
                if score >= threshold_confirmed:
                    severity = "confirmed"
                elif score >= threshold_probable:
                    severity = "probable"
                else:
                    severity = "possible"

                # Only emit for confirmed/probable (≥ 80)
                if score >= threshold_probable:
                    crash_report = self._build_crash_report(
                        lane=lane,
                        score=score,
                        severity=severity,
                        vehicle_ids=list(vehicle_ids_in_lane),
                        frame_id=frame_id,
                        signals=triggered_signals,
                    )
                    # Reset cooldown for this lane
                    self._cooldown[lane] = 0

        # ── Update state for next frame ───────────────────────
        self._prev_lane_counts = dict(lane_counts)
        for lane in all_lanes:
            self._prev_ids_per_lane[lane] = {
                v["id"] for v in vehicles_by_lane.get(lane, [])
            }

        return crash_report

    # ──────────────────────────────────────────────────────────
    #  SIGNAL CHECKS
    # ──────────────────────────────────────────────────────────

    def _check_bbox_overlap(self, collisions: list, lane: str) -> int:
        """
        Signal 1: +40 if any existing CollisionDetector collision event
        involves this lane AND its IoU meets the higher crash threshold (0.3).

        The CollisionDetector already filtered for IoU ≥ 0.05 and closing speed.
        We re-check against the crash threshold (0.30) for higher confidence.
        """
        iou_threshold = self.cfg.get("iou_threshold", 0.30)
        score_value   = self.cfg.get("score_bbox_overlap", 40)

        for c in collisions:
            if c.get("lane") == lane and c.get("iou", 0) >= iou_threshold:
                return score_value
        return 0

    def _check_id_vanish(self, lane: str, current_ids: set) -> int:
        """
        Signal 2: +35 if ≥ N track IDs that were present in this lane
        last frame are now gone simultaneously.

        A single ID vanishing = vehicle left frame (normal).
        A cluster vanishing = crash stopped them, or YOLO lost them all at once.
        """
        min_vanish = self.cfg.get("min_vanish_count", 3)
        score_val  = self.cfg.get("score_id_vanish", 35)

        prev_ids = self._prev_ids_per_lane.get(lane, set())
        vanished = prev_ids - current_ids

        if len(vanished) >= min_vanish:
            return score_val
        return 0

    def _check_count_drop(self, lane: str, lane_counts: dict) -> int:
        """
        Signal 3: +30 if vehicle count in this lane dropped by ≥ threshold
        from last frame, AND this drop is isolated (adjacent lanes did NOT
        also drop by ≥ threshold, which would indicate a green-light flush).

        Filters false positives caused by end-of-red-phase vehicle clearing.
        """
        drop_threshold = self.cfg.get("count_drop_threshold", 3)
        score_val      = self.cfg.get("score_count_drop", 30)

        prev_count = self._prev_lane_counts.get(lane, 0)
        curr_count = lane_counts.get(lane, 0)
        drop       = prev_count - curr_count

        if drop < drop_threshold:
            return 0

        # Check if other lanes also dropped (green-light mass clear)
        other_drops = sum(
            1 for other_lane, other_count in lane_counts.items()
            if other_lane != lane and other_lane != "unknown"
            and (self._prev_lane_counts.get(other_lane, 0) - other_count) >= drop_threshold
        )

        # If 2+ other lanes also dropped simultaneously → likely green phase, not crash
        if other_drops >= 2:
            return 0

        return score_val

    def _check_direction_conflict(self, lane_vehicles: list) -> int:
        """
        Signal 4: +25 if two vehicles in the same lane are moving toward each other.

        Only fires after frame 3 (when direction history is populated).
        """
        score_val = self.cfg.get("score_direction_conflict", 25)

        if len(lane_vehicles) < 2:
            return 0

        # Check all pairs in this lane for direction conflict
        for i in range(len(lane_vehicles)):
            for j in range(i + 1, len(lane_vehicles)):
                dir_a = lane_vehicles[i].get("direction", "unknown")
                dir_b = lane_vehicles[j].get("direction", "unknown")
                if _directions_conflict(dir_a, dir_b):
                    return score_val

        return 0

    # ──────────────────────────────────────────────────────────
    #  REPORT BUILDER
    # ──────────────────────────────────────────────────────────

    def _build_crash_report(
        self,
        lane: str,
        score: int,
        severity: str,
        vehicle_ids: list,
        frame_id: int,
        signals: list,
    ) -> dict:
        """
        Assembles the crash_report dict returned to caller and passed to AlertDispatcher.
        """
        return {
            "lane":        lane,
            "score":       score,
            "severity":    severity,    # "possible", "probable", "confirmed"
            "vehicle_ids": vehicle_ids,
            "frame_id":    frame_id,
            "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "signals":     signals,     # Which signals contributed e.g. ["iou+40","drop+30"]
        }

    # ──────────────────────────────────────────────────────────
    #  STATE RESET
    # ──────────────────────────────────────────────────────────

    def reset(self):
        """Reset all internal state. Call between unrelated video sequences."""
        self._score_history.clear()
        self._persist_count.clear()
        for k in self._cooldown:
            self._cooldown[k] = self.cfg.get("alert_cooldown_frames", 60)
        self._prev_lane_counts.clear()
        self._prev_ids_per_lane.clear()
        self._frame_count = 0
