"""
example_usage.py
----------------
Demonstrates how to run the full traffic analysis pipeline.

Steps:
  1. Configure paths and parameters
  2. Run the pipeline (generator — no bulk memory allocation)
  3. Print per-frame structured output
  4. Summarise results after all frames are processed

Run:
    python example_usage.py
"""
import cv2
import os
import json
from pipeline import run_pipeline
from preprocessing import CONFIG as DEFAULT_PREPROCESS_CONFIG
from detector import DETECTOR_CONFIG
from tracker import TRACKER_CONFIG
from lane_mapper import LANE_CONFIG
from collision_detector import CollisionLogger

# ─────────────────────────────────────────────
#  1. USER CONFIGURATION — edit these
# ─────────────────────────────────────────────

# Path to your folder of extracted video frames (JPG/PNG files)
FRAMES_FOLDER = r"D:\Downloads\CrashFrame"   # ← Change this path

# Path for the collision log file (set to None to disable file logging)
COLLISION_LOG_FILE = None  # Logging to terminal only

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

# ── Lane boundaries ──────────────────────────
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

# ── Detector overrides ───────────────────────
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

# ── Tracker overrides ────────────────────────
CUSTOM_TRACKER_CONFIG = {
    **TRACKER_CONFIG,
    "max_distance":    80,               # Max pixels to match same vehicle
    "max_lost_frames": 10,               # Frames before track is dropped
    "min_hits":        1,                # Min frames to confirm a vehicle
}


# ─────────────────────────────────────────────
#  2. RESULT ACCUMULATOR
# ─────────────────────────────────────────────
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

        # ── Emergency vehicle tracking ───────────────────────────
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
        print("  PIPELINE SUMMARY")
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


# ─────────────────────────────────────────────
#  3. MAIN ENTRY POINT
# ─────────────────────────────────────────────
def main():
    # Validate frames folder
    if not os.path.isdir(FRAMES_FOLDER):
        print(f"[ERROR] Frames folder not found: {FRAMES_FOLDER}")
        print("        Please update FRAMES_FOLDER in example_usage.py")
        return

    print("=" * 55)
    print("  TRAFFIC ANALYSIS PIPELINE")
    print("=" * 55)
    print(f"  Source : {FRAMES_FOLDER}\n")

    summary          = PipelineSummary()
    collision_logger = CollisionLogger(log_file=COLLISION_LOG_FILE)

    # ── Run pipeline ─────────────────────────────────────────────
    # run_pipeline is a generator — frames are never stored in bulk
    for frame_output in run_pipeline(
        frames_folder=FRAMES_FOLDER,
        preprocess_config=PREPROCESS_CONFIG,
        detector_config=CUSTOM_DETECTOR_CONFIG,
        tracker_config=CUSTOM_TRACKER_CONFIG,
        lane_config=CUSTOM_LANE_CONFIG,
    ):
        # ── Display debug frame ──────────────────────────────────
        debug_frame = frame_output.get("debug_frame")

        if debug_frame is not None:
            debug_frame_bgr = cv2.cvtColor(debug_frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("Traffic View", debug_frame_bgr)

            if cv2.waitKey(1) & 0xFF == ord('q'):
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
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
