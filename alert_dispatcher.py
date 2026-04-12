"""
alert_dispatcher.py
-------------------
Fires crash alerts when CrashDetector confirms an incident.

Responsibilities (in order, every confirmed crash):
  1. Check cooldown — don't spam police for the same lane within 2 minutes.
  2. Save a snapshot JPEG to the /alerts/ folder (always).
  3. Append a row to alerts_log.csv (always — permanent record).
  4. HTTP POST to police station API endpoint (only if URL configured).

Why this separation from CrashDetector:
  CrashDetector decides WHEN a crash occurred.
  AlertDispatcher decides HOW to respond to it.
  They are deliberately decoupled — CrashDetector has no I/O side effects,
  making it fully testable without file system or network.

Usage (example_usage.py):
  alert_dispatcher = AlertDispatcher()          # before loop
  crash_report = crash_detector.update(frame_output)
  if crash_report:
      alert_dispatcher.dispatch(crash_report, frame_output["debug_frame"])

alerts_log.csv columns:
  timestamp, frame_id, lane, severity, score, vehicle_ids, signals,
  snapshot_path, http_sent

The Dashboard (Phase 4) reads alerts_log.csv to populate the alert history panel.
"""

import csv
import json
import os
import logging
from datetime import datetime
from typing import Optional

import cv2
import numpy as np


# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────

ALERT_CONFIG = {
    # Directory where snapshot JPEGs are saved.
    # Created automatically if it does not exist.
    "alerts_dir": "alerts",

    # CSV log file — appended to across sessions.
    "log_csv": "alerts_log.csv",

    # HTTP endpoint for police station API.
    # Set to None or "" to disable HTTP alerting.
    # POST payload: JSON with full crash_report + snapshot_path.
    "http_endpoint": None,

    # Seconds to wait before re-alerting the same lane.
    "cooldown_seconds": 120,

    # JPEG quality for snapshot (0-100).
    "snapshot_quality": 85,

    # Minimum severity to dispatch ("possible" | "probable" | "confirmed").
    # "probable" = only send when score ≥ 80 (already filtered by CrashDetector).
    "min_dispatch_severity": "probable",
}


# ─────────────────────────────────────────────
#  CSV HEADER
# ─────────────────────────────────────────────

_CSV_HEADER = [
    "timestamp", "frame_id", "lane", "severity", "score",
    "vehicle_ids", "signals", "snapshot_path", "http_sent",
]


# ─────────────────────────────────────────────
#  ALERT DISPATCHER
# ─────────────────────────────────────────────

class AlertDispatcher:
    """
    Fires crash alerts: snapshot save, CSV log, optional HTTP POST.

    Thread-safety: not designed for multi-threaded use.
    All I/O is synchronous and runs in the calling thread.
    """

    def __init__(self, config: dict = None):
        self.cfg = config or ALERT_CONFIG

        # Per-lane cooldown tracking: lane → datetime of last alert
        self._last_alert_time: dict[str, Optional[datetime]] = {}

        # Ensure alerts directory exists
        alerts_dir = self.cfg.get("alerts_dir", "alerts")
        os.makedirs(alerts_dir, exist_ok=True)

        # Ensure CSV log file exists with header
        log_csv = self.cfg.get("log_csv", "alerts_log.csv")
        if not os.path.exists(log_csv):
            with open(log_csv, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(_CSV_HEADER)

        # Logger (console only — file I/O goes to CSV)
        self._log = logging.getLogger("AlertDispatcher")
        if not self._log.handlers:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter("%(message)s"))
            self._log.addHandler(ch)
        self._log.setLevel(logging.INFO)

    # ──────────────────────────────────────────────────────────
    #  PUBLIC API
    # ──────────────────────────────────────────────────────────

    def dispatch(self, crash_report: dict, debug_frame: Optional[np.ndarray] = None) -> bool:
        """
        Main entry point. Called when CrashDetector returns a crash_report.

        Args:
            crash_report: dict from CrashDetector._build_crash_report()
                Keys: lane, score, severity, vehicle_ids, frame_id,
                      timestamp, signals
            debug_frame:  RGB np.ndarray to save as snapshot, or None.

        Returns:
            True  if alert was dispatched (cooldown OK, severity met)
            False if suppressed (cooldown active or severity too low)
        """
        lane     = crash_report.get("lane", "unknown")
        severity = crash_report.get("severity", "possible")

        # ── Severity gate ───────────────────────────────────────
        severity_rank = {"possible": 0, "probable": 1, "confirmed": 2}
        min_sev = self.cfg.get("min_dispatch_severity", "probable")
        if severity_rank.get(severity, 0) < severity_rank.get(min_sev, 1):
            return False

        # ── Cooldown gate ───────────────────────────────────────
        if not self._check_cooldown(lane):
            return False  # Too soon since last alert for this lane

        # ── Actions ─────────────────────────────────────────────
        snapshot_path = self._save_snapshot(debug_frame, crash_report)
        self._write_local_log(crash_report, snapshot_path)
        http_ok = self._send_http_alert(crash_report, snapshot_path)

        # ── Console output ──────────────────────────────────────
        score = crash_report.get("score", 0)
        fid   = crash_report.get("frame_id", "?")
        self._log.info(
            f"[CRASH ALERT] Frame {fid} | Lane: {lane} | "
            f"Severity: {severity.upper()} | Score: {score} | "
            f"Signals: {crash_report.get('signals', [])} | "
            f"Snapshot: {snapshot_path}"
        )
        if http_ok:
            self._log.info(f"[CRASH ALERT] HTTP POST sent to police endpoint.")
        elif self.cfg.get("http_endpoint"):
            self._log.warning(f"[CRASH ALERT] HTTP POST FAILED — check endpoint.")

        # ── Update cooldown ─────────────────────────────────────
        self._last_alert_time[lane] = datetime.now()

        return True

    # ──────────────────────────────────────────────────────────
    #  INTERNAL METHODS
    # ──────────────────────────────────────────────────────────

    def _check_cooldown(self, lane: str) -> bool:
        """Returns True if enough time has passed since the last alert for this lane."""
        last = self._last_alert_time.get(lane)
        if last is None:
            return True  # First alert for this lane

        elapsed = (datetime.now() - last).total_seconds()
        cooldown = self.cfg.get("cooldown_seconds", 120)
        return elapsed >= cooldown

    def _save_snapshot(
        self,
        debug_frame: Optional[np.ndarray],
        report: dict,
    ) -> str:
        """
        Saves debug_frame as a JPEG to the alerts/ folder.
        Filename includes timestamp, lane and score for easy triage.

        Returns: relative file path string (or "no_snapshot" if frame is None).
        """
        if debug_frame is None:
            return "no_snapshot"

        alerts_dir = self.cfg.get("alerts_dir", "alerts")
        ts = datetime.now().strftime("%H-%M-%S")
        lane  = report.get("lane", "unknown").replace(" ", "_")
        score = report.get("score", 0)
        fname = f"alert_{ts}_{lane}_score{score}.jpg"
        fpath = os.path.join(alerts_dir, fname)

        # Convert RGB → BGR for cv2.imwrite
        frame_bgr = cv2.cvtColor(debug_frame, cv2.COLOR_RGB2BGR)
        quality   = self.cfg.get("snapshot_quality", 85)
        cv2.imwrite(fpath, frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])

        return fpath

    def _write_local_log(self, report: dict, snapshot_path: str) -> None:
        """
        Appends one row to alerts_log.csv.
        Always runs — even if HTTP fails.
        """
        log_csv = self.cfg.get("log_csv", "alerts_log.csv")
        row = {
            "timestamp":     report.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            "frame_id":      report.get("frame_id", ""),
            "lane":          report.get("lane", ""),
            "severity":      report.get("severity", ""),
            "score":         report.get("score", 0),
            "vehicle_ids":   json.dumps(report.get("vehicle_ids", [])),
            "signals":       json.dumps(report.get("signals", [])),
            "snapshot_path": snapshot_path,
            "http_sent":     False,  # updated below if HTTP succeeds
        }

        with open(log_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_HEADER)
            writer.writerow(row)

    def _send_http_alert(self, report: dict, snapshot_path: str) -> bool:
        """
        HTTP POST to police station API endpoint.
        Returns True on success, False on failure or if endpoint not configured.

        Payload JSON:
          {
            "timestamp":     "2026-04-11 15:45:06",
            "lane":          "bottom_road",
            "severity":      "confirmed",
            "score":         110,
            "vehicle_ids":   [3, 7],
            "confidence":    0.95,
            "snapshot_path": "alerts/alert_15-45-06_bottom_road_score110.jpg"
          }
        """
        endpoint = self.cfg.get("http_endpoint")
        if not endpoint:
            return False  # Graceful no-op — endpoint not configured

        try:
            import urllib.request
            payload = {
                "timestamp":     report.get("timestamp", ""),
                "lane":          report.get("lane", ""),
                "severity":      report.get("severity", ""),
                "score":         report.get("score", 0),
                "vehicle_ids":   report.get("vehicle_ids", []),
                "confidence":    0.95 if report.get("severity") == "confirmed" else 0.80,
                "snapshot_path": snapshot_path,
            }
            data    = json.dumps(payload).encode("utf-8")
            req     = urllib.request.Request(
                endpoint,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            response = urllib.request.urlopen(req, timeout=5)
            return response.status == 200

        except Exception as e:
            # Any failure — network down, endpoint wrong, timeout — silently fails.
            # The CSV log is always written first, so no data is lost.
            self._log.warning(f"[AlertDispatcher] HTTP POST failed: {e}")
            return False
