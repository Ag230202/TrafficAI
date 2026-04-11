"""
data_logger.py
--------------
Silently records per-frame state vectors to traffic_log.csv during
Phase 2 operation. This CSV becomes the training dataset for the
Phase 3 DQN neural network.

Collection happens with zero performance impact — one CSV row appended
per processed frame, containing a 10-element state vector,
the current signal action (which lane got green), and an estimated reward.

State vector (10 numbers — DQN input):
  [count_left, count_bottom, count_right, count_top,
   wait_left, wait_bottom, wait_right, wait_top,
   current_phase, time_in_phase]

Action (int 0-3):
  Which phase the rule-based controller is currently running.
  0 = North-South (top+bottom)
  1 = East (right)
  2 = North-South again (second rotation)
  3 = West (left)

Reward (float):
  Approximated from lane_counts.
  -(vehicles waiting this frame) = negative means bad (high queue).
  Vehicles waiting = total vehicles not in the active phase lanes.
  Simple proxy — real wait time not measured per vehicle.

Collection target: run during Phase 2 production for at least 1 week
to capture morning rush, evening rush, night and weekend patterns.
At frame_skip=3 and 30fps = ~10 frames/second of real time processed
= ~864,000 rows/day. Manageable CSV (< 200MB/day).

Usage:
    data_logger = DataLogger()                    # before loop
    data_logger.log(frame_output, signal_output)  # inside loop (1 line)
    data_logger.close()                           # after loop (flush)
"""

import csv
import os
from datetime import datetime
from typing import Optional


# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────

DATA_LOGGER_CONFIG = {
    "log_path":   "traffic_log.csv",   # Output CSV file
    "flush_every": 100,                # Flush to disk every N rows (performance)
    "lanes": [                         # Canonical lane order — must match DQN state vector
        "left_road",
        "bottom_road",
        "right_road",
        "top_road",
    ],
}

_CSV_HEADER = [
    "timestamp",
    "frame_id",
    # State vector
    "count_left", "count_bottom", "count_right", "count_top",
    "wait_left",  "wait_bottom",  "wait_right",  "wait_top",
    "current_phase",
    "time_in_phase",
    # Action and reward
    "action",
    "reward",
]


# ─────────────────────────────────────────────
#  DATA LOGGER
# ─────────────────────────────────────────────

class DataLogger:
    """
    Logs per-frame state vectors to CSV for offline DQN training (Phase 3).

    Call log() once per frame inside the pipeline loop.
    Call close() after the loop to flush the buffer.
    """

    def __init__(self, config: dict = None):
        self.cfg    = config or DATA_LOGGER_CONFIG
        self._lanes = self.cfg.get("lanes", [
            "left_road", "bottom_road", "right_road", "top_road"
        ])
        self._rows_written = 0
        self._flush_every  = self.cfg.get("flush_every", 100)

        log_path = self.cfg.get("log_path", "traffic_log.csv")

        # Track cumulative wait cycles per lane (how many frames a lane had 0 green)
        self._wait_frames: dict[str, int] = {lane: 0 for lane in self._lanes}

        # Open CSV (append mode — survives multiple sessions)
        write_header = not os.path.exists(log_path)
        self._file   = open(log_path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=_CSV_HEADER)
        if write_header:
            self._writer.writeheader()

    # ──────────────────────────────────────────────────────────
    #  PUBLIC API
    # ──────────────────────────────────────────────────────────

    def log(self, frame_output: dict, signal_output) -> None:
        """
        Record one row to the CSV for the current frame.

        Args:
            frame_output:  dict from run_pipeline() generator
            signal_output: SignalPhaseOutput from SignalController.update()
        """
        lane_counts = frame_output.get("lane_counts", {})
        frame_id    = frame_output.get("frame_id", 0)

        # ── State vector ─────────────────────────────────────
        counts = [lane_counts.get(lane, 0) for lane in self._lanes]

        # Update wait frame counters
        active_lanes = set(signal_output.active_lanes) if signal_output else set()
        for lane in self._lanes:
            if lane not in active_lanes:
                self._wait_frames[lane] += 1
            else:
                self._wait_frames[lane] = 0  # Reset when lane gets green

        waits = [self._wait_frames[lane] for lane in self._lanes]

        # Phase info
        current_phase  = signal_output.phase_id if signal_output else -1
        time_in_phase  = round(signal_output.elapsed_in_phase, 2) if signal_output else 0.0

        # ── Action = current phase id (0-3) ──────────────────
        action = current_phase

        # ── Reward = -(vehicles not served this frame) ───────
        # Vehicles in non-active lanes are "waiting"
        if signal_output and signal_output.active_lanes:
            served_count  = sum(lane_counts.get(l, 0) for l in signal_output.active_lanes)
            waiting_count = sum(lane_counts.values()) - served_count
        else:
            waiting_count = sum(lane_counts.values())

        reward = -float(waiting_count)

        # ── Write row ────────────────────────────────────────
        row = {
            "timestamp":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "frame_id":      frame_id,
            "count_left":    counts[0],
            "count_bottom":  counts[1],
            "count_right":   counts[2],
            "count_top":     counts[3],
            "wait_left":     waits[0],
            "wait_bottom":   waits[1],
            "wait_right":    waits[2],
            "wait_top":      waits[3],
            "current_phase": current_phase,
            "time_in_phase": time_in_phase,
            "action":        action,
            "reward":        reward,
        }
        self._writer.writerow(row)
        self._rows_written += 1

        if self._rows_written % self._flush_every == 0:
            self._file.flush()

    def close(self) -> None:
        """Flush and close the CSV file. Call after the pipeline loop ends."""
        if self._file and not self._file.closed:
            self._file.flush()
            self._file.close()
        print(f"[DataLogger] Closed — {self._rows_written} rows written to "
              f"{self.cfg.get('log_path', 'traffic_log.csv')}")

    def get_rows_written(self) -> int:
        """Returns total rows written in this session."""
        return self._rows_written
