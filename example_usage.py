"""
example_usage.py
----------------------------
Entrypoint for the full integrated Traffic AI pipeline:
  Phase 1 — YOLO detection + centroid tracking + lane mapping + emergency detection
  Phase 2 — Crash confidence scoring + police alert dispatch + rule-based signal timing
            + data logging for Phase 3 DQN training

Steps:
  1. Configure paths and parameters
  2. Initialise Phase 1 pipeline + Phase 2 components
  3. Process frames: detect → track → crash score → signal control → alert → log
  4. Print summaries for traffic, collisions and signal timing

Run:
    python example_usage.py

Dependencies:
  - Phase 1: pipeline.py, detector.py, tracker.py, lane_mapper.py, emergency_detector.py
  - Phase 2: crash_detector.py, alert_dispatcher.py, signal_controller.py,
             signal_logger.py, data_logger.py
"""

import cv2
import os
import json
import numpy as np
from pipeline import run_pipeline
from preprocessing import CONFIG as DEFAULT_PREPROCESS_CONFIG
from detector import DETECTOR_CONFIG
from tracker import TRACKER_CONFIG
from lane_mapper import LANE_CONFIG
from collision_detector import CollisionLogger

# ── Phase 2 imports ──────────────────────────────────────────────
from signal_controller import SignalController, SIGNAL_CONFIG
from signal_logger import SignalLogger
from crash_detector import CrashDetector
from alert_dispatcher import AlertDispatcher
from data_logger import DataLogger


# ═════════════════════════════════════════════════════════════════
#  1. USER CONFIGURATION — edit these
# ═════════════════════════════════════════════════════════════════

# Path to your folder of extracted video frames (JPG/PNG files)
FRAMES_FOLDER = r"D:\Traffic_AI\Traffic_Footage_Sanity"   # ← Change this path

# ── Preprocessing overrides ─────────────────
PREPROCESS_CONFIG = {
    **DEFAULT_PREPROCESS_CONFIG,         # Start from defaults
    "resize_width":  640,
    "resize_height": 480,
    "frame_skip":    3,                  # Process every 3rd frame

    # FIX 3: ROI disabled for top-down intersection camera — the whole
    # frame is road. Previous ROI (y=120) was cutting police cars,
    # ambulance, and background vehicles from detection entirely.
    "rois": [],

    "alpha": 1.2,                        # Contrast boost (used when use_clahe=False)
    "beta":  15,                         # Brightness boost (used when use_clahe=False)
    "blur_kernel": (3, 3),               # Mild noise reduction

    # FIX 1: CLAHE enabled for night-time / mixed lighting footage.
    # Unlike alpha/beta (global transform), CLAHE equalises contrast
    # locally per tile — handles blown-out emergency lights AND dark road
    # areas simultaneously. Set to False for daytime footage.
    "use_clahe":        True,
    "clahe_clip_limit": 2.0,             # Higher = more contrast, more noise risk
    "clahe_tile_grid":  (8, 8),          # Tile size for 640x480 frames

    "use_background_subtraction": False, # Toggle MOG2 background subtraction
}

# ─ Lane boundaries ───────────────────────────────────────────────
CUSTOM_LANE_CONFIG = {
    "left_road": [
        (0, 240), (240, 180), (220, 480), (0, 480)
    ],

    "bottom_road": [
        (180, 280), (460, 280), (460, 480), (180, 480)
    ],

    "right_road": [
        (420, 200), (640, 160), (640, 480), (420, 480)
    ],

    "top_road": [
        (180, 0), (460, 0), (460, 260), (240, 180)
    ],
}

# ─ Detector overrides ────────────────────────────────────────────
CUSTOM_DETECTOR_CONFIG = {
    **DETECTOR_CONFIG,
    "confidence_threshold": 0.15,        # FIX 1: lowered from 0.25 — night footage
                                         # drops confidence significantly; 0.15 catches
                                         # fire trucks, police cars, distant vehicles
                                         # that were previously filtered out.

    # FIX 2: increased from 640 to 1280 for overhead/top-down camera.
    # At 640, vehicles in the upper half of the frame are ~20-40px wide
    # and fall between YOLO grid cells. 1280 doubles grid density, allowing
    # detection of vehicles half the size. ~2x slower but essential here.
    "imgsz": 1280,
}

# ─ Tracker overrides ─────────────────────────────────────────────
CUSTOM_TRACKER_CONFIG = {
    **TRACKER_CONFIG,
    "max_distance":    80,               # Max pixels to match same vehicle
    "max_lost_frames": 10,               # Frames before track is dropped
    "min_hits":        1,                # Min frames to confirm a vehicle
}

# ─ Signal controller overrides (Phase 2) ────────────────────────
# Start from default SIGNAL_CONFIG and customize as needed
CUSTOM_SIGNAL_CONFIG = {
    **SIGNAL_CONFIG,
    # Timing overrides (seconds)
    "cycle_duration":           60,      # Total seconds per cycle
    "base_green_duration":      20,      # Starting green
    "min_green_duration":       8,       # Minimum (pedestrians)
    "max_green_duration":       50,      # Maximum
    "yellow_duration":          4,       # Transition time
    
    # Emergency settings
    "emergency_duration":       25,      # Green for ambulances/fire trucks
    "emergency_preemption":     True,    # Enable emergency priority
    
    # Density scaling
    "density_scaling_factor":   0.8,     # 0.0-1.0 (higher = more responsive)
    "enable_adaptive":          True,    # Scale green by demand
    "use_dqn":                  False,    # Enable DQN
    
    # Collision handling
    "collision_red_timeout":    5,       # Frames to keep red after crash
    "enable_collision_override": True,   # Force red on collision lanes
    
    # Debug
    "debug_mode":               False,   # Print per-frame info (verbose)
}

# ─ Output file paths ────────────────────────────────────────────
COLLISION_LOG_FILE = "collision_log.txt"     # Collision event log
SIGNAL_LOG_FILE    = "signal_log.txt"        # Signal phase log

# ─ Phase 2 toggle ───────────────────────────────────────────────
ENABLE_SIGNAL_CONTROLLER = True              # Set False to run Phase 1 only


# ═════════════════════════════════════════════════════════════════
#  2. RESULT ACCUMULATORS
# ═════════════════════════════════════════════════════════════════

class PipelineSummary:
    """Tracks aggregate stats across all frames without storing raw data."""

    def __init__(self):
        self.total_frames          = 0
        self.total_vehicles        = 0
        self.emergency_frames      = []   # frame_ids where emergency detected
        self.lane_totals           = {}   # cumulative vehicle counts per lane
        self.direction_counts      = {}   # direction → count

        # Emergency vehicle tracking
        self.total_emergency_detections = 0          # total emergency flags across all frames
        self.emergency_lane_counts      = {}         # lane → how many frames had emergency there
        self.max_emergency_per_frame    = 0          # peak simultaneous emergency lane count
        self.emergency_vehicle_ids      = set()      # unique track IDs flagged as emergency

    def update(self, frame_output: dict):
        self.total_frames   += 1
        self.total_vehicles += len(frame_output["vehicles"])

        # Accumulate lane counts
        for lane, count in frame_output["lane_counts"].items():
            self.lane_totals[lane] = self.lane_totals.get(lane, 0) + count

        # Accumulate direction counts
        for v in frame_output["vehicles"]:
            d = v.get("direction", "unknown")
            self.direction_counts[d] = self.direction_counts.get(d, 0) + 1

        # ── Emergency vehicle tracking ───────────────────────────────
        emerg_lanes = frame_output.get("emergency_lane", [])

        if emerg_lanes:
            fid = frame_output["frame_id"]

            # Record frame ID (deduplicated)
            if fid not in self.emergency_frames:
                self.emergency_frames.append(fid)

            # Count detections this frame
            self.total_emergency_detections += len(emerg_lanes)

            # Track which lanes had emergency vehicles and how often
            for lane in emerg_lanes:
                self.emergency_lane_counts[lane] = (
                    self.emergency_lane_counts.get(lane, 0) + 1
                )

            # Track peak simultaneous emergency lanes
            self.max_emergency_per_frame = max(
                self.max_emergency_per_frame, len(emerg_lanes)
            )

            # Track unique vehicle IDs that were in emergency lanes
            for vid in frame_output.get("emergency_veh_ids", set()):
                self.emergency_vehicle_ids.add(vid)

    def print_summary(self):
        print("\n" + "═" * 55)
        print("  PIPELINE SUMMARY (Phase 1: Traffic Analysis)")
        print("═" * 55)
        print(f"  Frames processed        : {self.total_frames}")
        print(f"  Total vehicle obs       : {self.total_vehicles}")
        print()

        # ── Emergency section ────────────────────────────────────
        print(f"  Emergency events")
        print(f"  ├─ Frames with emergency : {len(self.emergency_frames)}")
        if self.emergency_frames:
            print(f"  ├─ Frame IDs             : {self.emergency_frames}")
        print(f"  ├─ Total detections      : {self.total_emergency_detections}")
        print(f"  ├─ Peak lanes at once    : {self.max_emergency_per_frame}")
        id_str = f"  {sorted(self.emergency_vehicle_ids)}" if self.emergency_vehicle_ids else "  none"
        print(f"  └─ Emergency vehicle IDs : {len(self.emergency_vehicle_ids)}{id_str}")
        if self.emergency_lane_counts:
            print()
            print("  Emergency lane breakdown:")
            for lane, count in sorted(self.emergency_lane_counts.items(),
                                      key=lambda x: -x[1]):
                bar = "█" * min(count, 30)
                print(f"    {lane:<12} {count:>4} frames  {bar}")
        print()

        # ── Lane counts ──────────────────────────────────────────
        print("  Cumulative lane counts:")
        for lane, count in sorted(self.lane_totals.items()):
            bar = "█" * min(count, 40)
            print(f"    {lane:<10} {count:>5}  {bar}")
        print()

        # ── Directions ───────────────────────────────────────────
        print("  Vehicle directions:")
        for direction, count in sorted(self.direction_counts.items(),
                                       key=lambda x: -x[1]):
            print(f"    {direction:<12} {count}")
        print("═" * 55)


# ═════════════════════════════════════════════════════════════════
#  3. VISUALIZATION HELPERS (Phase 2)
# ═════════════════════════════════════════════════════════════════

def draw_signal_state(debug_frame, signal_output, lane_mapper):
    """
    Draw traffic signal indicators on the debug frame.
    
    Args:
        debug_frame: RGB image to draw on
        signal_output: SignalPhaseOutput from controller.update()
        lane_mapper: LaneMapper instance with lane boundaries
    """
    lane_boundaries = lane_mapper.get_lane_boundaries()
    
    # Color coding (RGB)
    colors = {
        "green":  (0, 255, 0),       # Green
        "yellow": (0, 255, 255),     # Yellow
        "red":    (255, 0, 0),       # Red
    }
    
    # Draw lane signal indicators
    for lane_name, polygon in lane_boundaries.items():
        if lane_name == "unknown":
            continue
        
        # Determine signal color
        if lane_name in signal_output.active_lanes:
            signal_color = colors["green"]
            signal_text = "GREEN"
        elif lane_name in signal_output.yellow_lanes:
            signal_color = colors["yellow"]
            signal_text = "YELLOW"
        elif lane_name in signal_output.red_lanes:
            signal_color = colors["red"]
            signal_text = "RED"
        else:
            signal_color = (100, 100, 100)  # Gray (unknown)
            signal_text = "???"
        
        # Draw label at lane corner
        pts = np.array(polygon, dtype=np.int32)
        label_x, label_y = pts[0]
        
        # Draw signal box
        box_size = 40
        cv2.rectangle(
            debug_frame,
            (label_x - 20, label_y - box_size - 10),
            (label_x + 20, label_y - 10),
            signal_color, -1
        )
        
        cv2.putText(
            debug_frame,
            signal_text,
            (label_x - 15, label_y - 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
        )
        
        # Draw lane name
        cv2.putText(
            debug_frame,
            f"{lane_name}",
            (label_x - 30, label_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1
        )
    
    # Draw overall phase info at top
    phase_info = (
        f"Phase {signal_output.phase_id}: {signal_output.phase_name} | "
        f"Green: {signal_output.green_duration}s | "
        f"Elapsed: {signal_output.elapsed_in_phase:.1f}s"
    )
    
    cv2.putText(
        debug_frame,
        phase_info,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
    )
    
    # Draw override reason if applicable
    if signal_output.override_reason and "standard" not in signal_output.override_reason:
        override_text = f"⚠️ {signal_output.override_reason.upper()}"
        cv2.putText(
            debug_frame,
            override_text,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2
        )
    
    return debug_frame


# ═════════════════════════════════════════════════════════════════
#  4. MAIN ENTRY POINT
# ═════════════════════════════════════════════════════════════════

def main():
    """Main pipeline execution with Phase 1 (traffic analysis) + Phase 2 (signal control)."""
    
    # Validate frames folder
    if not os.path.isdir(FRAMES_FOLDER):
        print(f"[ERROR] Frames folder not found: {FRAMES_FOLDER}")
        print("        Please update FRAMES_FOLDER in example_usage_integrated.py")
        return

    print("=" * 55)
    print("  TRAFFIC ANALYSIS PIPELINE")
    print("=" * 55)
    print(f"  Source : {FRAMES_FOLDER}\n")

    summary          = PipelineSummary()
    collision_logger = CollisionLogger(log_file=COLLISION_LOG_FILE)

    # ── Phase 2 initialisation ───────────────────────────────────
    signal_controller = None
    signal_logger     = None
    lane_mapper_ref   = None
    crash_detector    = CrashDetector()
    alert_dispatcher  = AlertDispatcher()
    data_logger       = DataLogger()

    if ENABLE_SIGNAL_CONTROLLER:
        signal_controller = SignalController(config=CUSTOM_SIGNAL_CONFIG)
        signal_logger     = SignalLogger(log_file=SIGNAL_LOG_FILE)
        from lane_mapper import LaneMapper
        lane_mapper_ref   = LaneMapper(CUSTOM_LANE_CONFIG)

    # ── Run pipeline ─────────────────────────────────────────────
    # run_pipeline is a generator — frames are never stored in bulk
    frame_count = 0
    
    for frame_output in run_pipeline(
        frames_folder=FRAMES_FOLDER,
        preprocess_config=PREPROCESS_CONFIG,
        detector_config=CUSTOM_DETECTOR_CONFIG,
        tracker_config=CUSTOM_TRACKER_CONFIG,
        lane_config=CUSTOM_LANE_CONFIG,
    ):
        frame_count += 1
        fid = frame_output["frame_id"]
        
        # ── Phase 2: Crash Detection (confidence-scored, temporal persistence) ──
        # Runs FIRST — output gates the signal red override below.
        crash_report = crash_detector.update(frame_output)
        if crash_report:
            sev   = crash_report["severity"].upper()
            score = crash_report["score"]
            lane  = crash_report["lane"]
            print(f"  Frame {fid:>5} | [CRASH {sev}] score={score} "
                  f"lane={lane} signals={crash_report['signals']}")
            alert_dispatcher.dispatch(crash_report, frame_output.get("debug_frame"))

        # ── Phase 2: Signal Control ───────────────────────────────────────
        # Only pass a collision to signal_controller when crash_detector
        # has CONFIRMED one (score >= 80, 3 consecutive frames).
        # This replaces the raw noisy IoU collision list — eliminates
        # the 59 false red-overrides seen when using raw collisions.
        if ENABLE_SIGNAL_CONTROLLER and signal_controller:
            gated_collisions = [{"lane": crash_report["lane"]}] if crash_report else []
            signal_output = signal_controller.update(
                lane_counts=frame_output["lane_counts"],
                emergency_lane=frame_output["emergency_lane"],
                collisions=gated_collisions,
                frame_id=fid
            )
            signal_logger.log(fid, signal_output)
        else:
            signal_output = None

        # ── Phase 2: Data logging for Phase 3 DQN training ───────────────
        data_logger.log(frame_output, signal_output)

        # ── Display debug frame with signal overlay ──────────────────
        debug_frame = frame_output.get("debug_frame")

        if debug_frame is not None:
            # Draw signal state overlay if Phase 2 is enabled
            if ENABLE_SIGNAL_CONTROLLER and signal_output and lane_mapper_ref:
                debug_frame = draw_signal_state(debug_frame, signal_output, lane_mapper_ref)
            
            # Convert RGB to BGR for OpenCV display
            debug_frame_bgr = cv2.cvtColor(debug_frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("Traffic View (Phase 1 + Phase 2)", debug_frame_bgr)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[User Exit] Quit requested")
                break

        # ── Per-frame console output (short mode) ──────────────────
        fid        = frame_output["frame_id"]
        vehicles   = frame_output["vehicles"]
        emerg_lane = frame_output["emergency_lane"]
        emerg_ids  = frame_output.get("emergency_veh_ids", set())

        if emerg_lane:
            print(f"  ⚠  Frame {fid:>5} | EMERGENCY → {emerg_lane} "
                  f"| vehicle IDs {sorted(emerg_ids)}")


        # ── Collision logging ────────────────────────────────────
        # CollisionLogger prints [ACCIDENT] lines and updates stats.
        collision_logger.log(frame_output.get("collisions", []))

        summary.update(frame_output)

    # ── End-of-run summaries ─────────────────────────────────────
    summary.print_summary()
    collision_logger.print_summary()
    if signal_logger:
        signal_logger.print_summary()
        signal_logger.export_json("signal_stats.json")
    data_logger.close()
    cv2.destroyAllWindows()
    print("\n[Complete] All summaries printed.")
    print("  • collision_log.txt   — per-collision events")
    print("  • signal_log.txt      — per-frame signal state")
    print("  • signal_stats.json   — signal summary statistics")
    print("  • traffic_log.csv     — Phase 3 DQN training data")
    print("  • alerts_log.csv      — confirmed crash alerts")
    print("  • alerts/             — crash snapshot images")


if __name__ == "__main__":
    main()
