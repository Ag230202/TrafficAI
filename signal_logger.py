"""
signal_logger.py
----------------
Logging and statistics tracking for traffic signal controller.

Responsibilities:
  - Log each frame's signal state to file and console
  - Track cumulative statistics (total green per lane, preemptions, etc.)
  - Provide summary output for PipelineSummary

Usage:
  logger = SignalLogger(log_file="signal_log.txt")
  for frame_output in run_pipeline(...):
      signal_output = signal_controller.update(...)
      logger.log(frame_output["frame_id"], signal_output)
  
  summary = logger.get_summary()
"""

import json
import logging
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from signal_controller import SignalPhaseOutput
from datetime import datetime


class SignalLogger:
    """
    Logs signal state per frame and accumulates statistics.
    """
    
    def __init__(self, log_file: Optional[str] = "signal_log.txt", debug: bool = False):
        self.log_file = log_file
        self.debug = debug
        
        # Initialize logger
        self.logger = logging.getLogger("SignalController")
        self.logger.setLevel(logging.INFO if not debug else logging.DEBUG)
        
        # File handler
        if log_file:
            fh = logging.FileHandler(log_file, mode='w')
            fh.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO if not debug else logging.DEBUG)
        formatter = logging.Formatter('%(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
        # Statistics accumulators
        self.stats = {
            "total_frames":            0,
            "frame_ids":               [],
            "phase_ids":               [],
            "green_per_lane":          defaultdict(list),      # lane → [green_durations]
            "active_lanes_per_frame":  [],
            "preemptions":             [],                    # {frame, reason, lanes}
            "collisions_detected":     [],                    # {frame, lanes_affected}
            "phase_transitions":       [],                    # {frame, from_phase, to_phase}
        }
        
        self._last_phase_id = None
        
    def log(self, frame_id: int, signal_output: SignalPhaseOutput) -> None:
        """
        Log signal state for one frame.
        
        Args:
            frame_id: frame number
            signal_output: SignalPhaseOutput from controller.update()
        """
        self.stats["total_frames"] += 1
        self.stats["frame_ids"].append(frame_id)
        self.stats["phase_ids"].append(signal_output.phase_id)
        
        # Track active lanes
        self.stats["active_lanes_per_frame"].append(signal_output.active_lanes)
        
        # Track green duration per lane
        for lane in signal_output.active_lanes:
            self.stats["green_per_lane"][lane].append(signal_output.green_duration)
        
        # Track preemptions (emergency or other overrides)
        if signal_output.override_reason == "emergency_preemption":
            self.stats["preemptions"].append({
                "frame": frame_id,
                "reason": signal_output.override_reason,
                "lanes": signal_output.active_lanes,
                "phase": signal_output.phase_id,
            })
        
        # Track collisions
        if signal_output.override_reason == "collision_red_override" and signal_output.red_lanes:
            self.stats["collisions_detected"].append({
                "frame": frame_id,
                "red_lanes": signal_output.red_lanes,
                "phase": signal_output.phase_id,
            })
        
        # Track phase transitions
        if signal_output.phase_id != self._last_phase_id:
            if self._last_phase_id is not None:
                self.stats["phase_transitions"].append({
                    "frame": frame_id,
                    "from_phase": self._last_phase_id,
                    "to_phase": signal_output.phase_id,
                    "phase_name": signal_output.phase_name,
                })
            self._last_phase_id = signal_output.phase_id
        
        # Log to file/console
        self._log_frame_detail(frame_id, signal_output)
    
    def _log_frame_detail(self, frame_id: int, signal_output: SignalPhaseOutput) -> None:
        """Log detailed per-frame information."""
        log_msg = (
            f"Frame {frame_id:5d} | "
            f"Phase {signal_output.phase_id} ({signal_output.phase_name:20s}) | "
            f"Green {int(signal_output.green_duration):2d}s | "
            f"Active: {', '.join(signal_output.active_lanes) or 'none':25s} | "
            f"Red: {', '.join(signal_output.red_lanes) or 'none':25s}"
        )
        
        if signal_output.override_reason and signal_output.override_reason != "standard_adaptive":
            log_msg += f" | Override: {signal_output.override_reason}"
        
        self.logger.info(log_msg)
    
    def get_summary(self) -> Dict:
        """
        Compute summary statistics across all logged frames.
        
        Returns:
            {
                "total_frames":           int,
                "avg_green_per_lane":     {lane: avg_seconds},
                "max_green_per_lane":     {lane: max_seconds},
                "total_preemptions":      int,
                "total_collisions":       int,
                "phase_distribution":     {phase_id: count},
                "avg_phase_duration":     float,
                "preemption_summary":     [...],
            }
        """
        total = self.stats["total_frames"]
        
        if total == 0:
            return {"error": "No frames logged yet"}
        
        # Compute averages and maxima
        avg_green_per_lane = {}
        max_green_per_lane = {}
        for lane, durations in self.stats["green_per_lane"].items():
            avg_green_per_lane[lane] = round(sum(durations) / len(durations), 2) if durations else 0
            max_green_per_lane[lane] = max(durations) if durations else 0
        
        # Phase distribution
        phase_dist = defaultdict(int)
        for phase_id in self.stats["phase_ids"]:
            phase_dist[phase_id] += 1
        
        # Preemption summary
        preemption_lanes = defaultdict(int)
        for preempt in self.stats["preemptions"]:
            for lane in preempt.get("lanes", []):
                preemption_lanes[lane] += 1
        
        # Phase duration (frames per phase)
        phase_durations = defaultdict(list)
        for i, phase_id in enumerate(self.stats["phase_ids"]):
            if i > 0 and self.stats["phase_ids"][i-1] != phase_id:
                # Phase just started
                phase_durations[phase_id].append(1)
            elif i > 0 and self.stats["phase_ids"][i-1] == phase_id:
                phase_durations[phase_id][-1] += 1
            else:
                phase_durations[phase_id].append(1)
        
        avg_phase_duration = (
            sum(len(pd) for pd in phase_durations.values()) / len(phase_durations)
            if phase_durations else 0
        )
        
        summary = {
            "total_frames":           total,
            "avg_green_per_lane":     avg_green_per_lane,
            "max_green_per_lane":     max_green_per_lane,
            "total_preemptions":      len(self.stats["preemptions"]),
            "preemptions_per_lane":   dict(preemption_lanes),
            "total_collisions":       len(self.stats["collisions_detected"]),
            "phase_distribution":     dict(phase_dist),
            "avg_phase_duration_frames": round(avg_phase_duration, 1),
            "phase_transitions_count": len(self.stats["phase_transitions"]),
        }
        
        return summary
    
    def print_summary(self) -> None:
        """Print human-readable summary to console."""
        summary = self.get_summary()
        
        print("\n" + "="*80)
        print("SIGNAL CONTROLLER SUMMARY")
        print("="*80)
        print(f"Total frames processed: {summary.get('total_frames', 0)}")
        print()
        
        print("Average Green Time per Lane:")
        for lane, avg_green in summary.get("avg_green_per_lane", {}).items():
            max_green = summary.get("max_green_per_lane", {}).get(lane, 0)
            print(f"  {lane:20s}: avg={avg_green:5.1f}s, max={int(max_green):2d}s")
        print()
        
        print("Phase Distribution:")
        for phase_id, count in sorted(summary.get("phase_distribution", {}).items()):
            pct = 100 * count / summary.get("total_frames", 1)
            print(f"  Phase {phase_id}: {count:4d} frames ({pct:5.1f}%)")
        print()
        
        print(f"Total Emergency Preemptions: {summary.get('total_preemptions', 0)}")
        if summary.get("preemptions_per_lane"):
            for lane, count in summary.get("preemptions_per_lane", {}).items():
                print(f"  {lane}: {count} preemptions")
        print()
        
        print(f"Total Collisions Detected (Red Overrides): {summary.get('total_collisions', 0)}")
        print(f"Phase Transitions: {summary.get('phase_transitions_count', 0)}")
        print(f"Avg Phase Duration: {summary.get('avg_phase_duration_frames', 0):.1f} frames")
        print("="*80 + "\n")
    
    def get_detailed_preemptions(self) -> List[Dict]:
        """Return detailed list of all preemptions."""
        return self.stats["preemptions"]
    
    def get_detailed_collisions(self) -> List[Dict]:
        """Return detailed list of all collision detections."""
        return self.stats["collisions_detected"]
    
    def export_json(self, filepath: str) -> None:
        """Export detailed statistics to JSON."""
        export_data = {
            "summary": self.get_summary(),
            "preemptions": self.stats["preemptions"],
            "collisions": self.stats["collisions_detected"],
            "phase_transitions": self.stats["phase_transitions"],
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"[SignalLogger] Exported detailed stats to {filepath}")