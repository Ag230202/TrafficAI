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
    "alerts_dir": "alerts",

    # CSV log file
    "log_csv": "alerts_log.csv",

    # ── TELEGRAM BOT CONFIGURATION ──
    "telegram_enabled": True,  # Set to True to enable Telegram alerts
    "telegram_bot_token": "8608991246:AAHI7jLPg8Oof483yxZKxBoCW_i-QhdSUzI",
    "telegram_chat_id": "929328064",

    # Seconds to wait before re-alerting the same lane.
    "cooldown_seconds": 120,

    # JPEG quality for snapshot (0-100).
    "snapshot_quality": 85,

    # Minimum severity to dispatch ("possible" | "probable" | "confirmed").
    "min_dispatch_severity": "possible",
}


# ─────────────────────────────────────────────
#  CSV HEADER
# ─────────────────────────────────────────────

_CSV_HEADER = [
    "timestamp", "frame_id", "lane", "severity", "score",
    "vehicle_ids", "signals", "snapshot_path", "telegram_sent",
]


# ─────────────────────────────────────────────
#  ALERT DISPATCHER
# ─────────────────────────────────────────────

class AlertDispatcher:
    """
    Fires crash alerts: snapshot save, CSV log, and Telegram message.
    """

    def __init__(self, config: dict = None):
        self.cfg = config or ALERT_CONFIG

        # Per-lane cooldown tracking
        self._last_alert_time: dict[str, Optional[datetime]] = {}

        # Ensure alerts directory exists
        alerts_dir = self.cfg.get("alerts_dir", "alerts")
        os.makedirs(alerts_dir, exist_ok=True)

        # Ensure CSV log file exists with header
        log_csv = self.cfg.get("log_csv", "alerts_log.csv")
        if not os.path.exists(log_csv):
            with open(log_csv, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(_CSV_HEADER)

        self._log = logging.getLogger("AlertDispatcher")
        if not self._log.handlers:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter("%(message)s"))
            self._log.addHandler(ch)
        self._log.setLevel(logging.INFO)

    def dispatch(self, crash_report: dict, debug_frame: Optional[np.ndarray] = None) -> bool:
        lane     = crash_report.get("lane", "unknown")
        severity = crash_report.get("severity", "possible")

        severity_rank = {"possible": 0, "probable": 1, "confirmed": 2}
        min_sev = self.cfg.get("min_dispatch_severity", "probable")
        if severity_rank.get(severity, 0) < severity_rank.get(min_sev, 1):
            return False

        if not self._check_cooldown(lane):
            return False

        snapshot_path = self._save_snapshot(debug_frame, crash_report)
        telegram_ok = self._send_telegram_alert(crash_report, snapshot_path)
        self._write_local_log(crash_report, snapshot_path, telegram_ok)

        score = crash_report.get("score", 0)
        fid   = crash_report.get("frame_id", "?")
        
        self._log.info(
            f"[CRASH ALERT] Frame {fid} | {lane} | {severity.upper()} | Score: {score}"
        )
        if telegram_ok:
            self._log.info(f"[TELEGRAM BOT] Alert and snapshot successfully delivered!")

        self._last_alert_time[lane] = datetime.now()
        return True

    def _check_cooldown(self, lane: str) -> bool:
        last = self._last_alert_time.get(lane)
        if last is None:
            return True
        elapsed = (datetime.now() - last).total_seconds()
        return elapsed >= self.cfg.get("cooldown_seconds", 120)

    def _save_snapshot(self, debug_frame: Optional[np.ndarray], report: dict) -> str:
        if debug_frame is None:
            return "no_snapshot"
        alerts_dir = self.cfg.get("alerts_dir", "alerts")
        ts = datetime.now().strftime("%H-%M-%S")
        lane  = report.get("lane", "unknown").replace(" ", "_")
        score = report.get("score", 0)
        fname = f"alert_{ts}_{lane}_score{score}.jpg"
        fpath = os.path.join(alerts_dir, fname)

        frame_bgr = cv2.cvtColor(debug_frame, cv2.COLOR_RGB2BGR)
        quality   = self.cfg.get("snapshot_quality", 85)
        cv2.imwrite(fpath, frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return fpath

    def _write_local_log(self, report: dict, snapshot_path: str, telegram_ok: bool) -> None:
        log_csv = self.cfg.get("log_csv", "alerts_log.csv")
        
        # FIX: Check if we need to update the header from 'http_sent' to 'telegram_sent'
        header_needs_update = False
        if os.path.exists(log_csv):
            with open(log_csv, 'r') as f:
                first_line = f.readline()
                if "http_sent" in first_line:
                    header_needs_update = True
        
        if header_needs_update:
            # Simple way to clear old log and start fresh with correct header
            with open(log_csv, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(_CSV_HEADER)

        row = {
            "timestamp":     report.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            "frame_id":      report.get("frame_id", ""),
            "lane":          report.get("lane", ""),
            "severity":      report.get("severity", ""),
            "score":         report.get("score", 0),
            "vehicle_ids":   json.dumps(report.get("vehicle_ids", [])),
            "signals":       json.dumps(report.get("signals", [])),
            "snapshot_path": snapshot_path,
            "telegram_sent": telegram_ok,
        }
        with open(log_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_HEADER)
            writer.writerow(row)

    def _send_telegram_alert(self, report: dict, snapshot_path: str) -> bool:
        if not self.cfg.get("telegram_enabled"):
            return False

        try:
            import requests
            
            token = self.cfg.get("telegram_bot_token")
            chat_id = self.cfg.get("telegram_chat_id")
            url = f"https://api.telegram.org/bot{token}/sendPhoto"

            # Format the caption using HTML (more robust than Markdown for underscores)
            caption = (
                f"🚨 <b>TRAFFIC AI ACCIDENT ALERT</b> 🚨\n\n"
                f"📍 <b>Location:</b> {report.get('lane', 'Unknown Lane')}\n"
                f"⚠️ <b>Severity:</b> {str(report.get('severity')).upper()}\n"
                f"⏱️ <b>Time:</b> {report.get('timestamp')}\n"
                f"🚗 <b>Vehicles Involved IDs:</b> {report.get('vehicle_ids')}\n\n"
                f"System Confidence Score: {report.get('score')} / 100"
            )

            files = {}
            if os.path.exists(snapshot_path):
                files = {'photo': open(snapshot_path, 'rb')}

            data = {
                'chat_id': chat_id,
                'caption': caption,
                'parse_mode': 'HTML'
            }

            if files:
                response = requests.post(url, data=data, files=files, timeout=10)
            else:
                # Fallback to text only if image is missing
                url_text = f"https://api.telegram.org/bot{token}/sendMessage"
                data_text = {'chat_id': chat_id, 'text': caption, 'parse_mode': 'HTML'}
                response = requests.post(url_text, data=data_text, timeout=5)

            if response.status_code != 200:
                self._log.warning(f"[TELEGRAM BOT] Telegram API Error {response.status_code}: {response.text}")
                return False

            return True

        except Exception as e:
            self._log.warning(f"[TELEGRAM BOT] System Error: {e}")
            return False
