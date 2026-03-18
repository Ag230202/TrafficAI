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

# ─────────────────────────────────────────────
#  1. USER CONFIGURATION — edit these
# ─────────────────────────────────────────────

# Path to your folder of extracted video frames (JPG/PNG files)
FRAMES_FOLDER = r"C:\Users\ANIMESH GARTIA\Downloads\Traffic_Footage_Demo"   # ← Change this path

# ── Preprocessing overrides ─────────────────
PREPROCESS_CONFIG = {
    **DEFAULT_PREPROCESS_CONFIG,         # Start from defaults
    "resize_width":  640,
    "resize_height": 480,
    "frame_skip":    3,                  # Process every 3rd frame

    # Road ROI — adjust to match your camera angle
    # Format: list of (x1, y1, x2, y2) rectangles
    "rois": [
        (0, 120, 640, 480),              # Crop out sky/horizon above y=120
    ],

    "alpha": 1.2,                        # Contrast boost
    "beta":  15,                         # Brightness boost
    "blur_kernel": (3, 3),               # Mild noise reduction
    "use_background_subtraction": False, # Toggle MOG2 background subtraction
}

# ── Lane boundaries ──────────────────────────
# Divide the 640px-wide frame into 4 equal vertical lanes
# Format: {"lane_name": (x1, y1, x2, y2)}
CUSTOM_LANE_CONFIG = {
    "lane_1": [(0, 215), (210, 215), (185, 480), (0, 480)],
    "lane_2": [(210, 180), (395, 180), (360, 480), (185, 480)],
    "lane_3": [(395, 140), (560, 140), (525, 480), (360, 480)],
    "lane_4": [(560, 140), (640, 140), (640, 480), (525, 480)],
}
# ── Detector overrides ───────────────────────
CUSTOM_DETECTOR_CONFIG = {
    **DETECTOR_CONFIG,
    "confidence_threshold": 0.25,         # Lower = more detections, more noise
}

# ── Tracker overrides ────────────────────────
CUSTOM_TRACKER_CONFIG = {
    **TRACKER_CONFIG,
    "max_distance":    80,               # Max pixels to match same vehicle
    "max_lost_frames": 10,               # Frames before track is dropped
    "min_hits":        2,                # Min frames to confirm a vehicle
}


# ─────────────────────────────────────────────
#  2. RESULT ACCUMULATOR
# ─────────────────────────────────────────────
class PipelineSummary:
    """Tracks aggregate stats across all frames without storing raw data."""

    def __init__(self):
        self.total_frames     = 0
        self.total_vehicles   = 0
        self.emergency_frames = []         # frame_ids with emergency vehicles
        self.lane_totals      = {}         # cumulative vehicle counts per lane
        self.direction_counts = {}         # direction → count

    def update(self, frame_output: dict):
        self.total_frames   += 1
        self.total_vehicles += len(frame_output["vehicles"])

        # Accumulate lane counts
        for lane, count in frame_output["lane_counts"].items():
            self.lane_totals[lane] = self.lane_totals.get(lane, 0) + count

        # Track emergency events
        if frame_output["emergency_lane"]:
            self.emergency_frames.append(frame_output["frame_id"])

        # Accumulate direction counts
        for v in frame_output["vehicles"]:
            d = v.get("direction", "unknown")
            self.direction_counts[d] = self.direction_counts.get(d, 0) + 1

    def print_summary(self):
        print("\n" + "═" * 55)
        print("  PIPELINE SUMMARY")
        print("═" * 55)
        print(f"  Frames processed   : {self.total_frames}")
        print(f"  Total vehicle obs  : {self.total_vehicles}")
        print(f"  Emergency events   : {len(self.emergency_frames)}")
        if self.emergency_frames:
            print(f"  Emergency frame IDs: {self.emergency_frames}")
        print()
        print("  Cumulative lane counts:")
        for lane, count in sorted(self.lane_totals.items()):
            bar = "█" * min(count, 40)
            print(f"    {lane:<10} {count:>5}  {bar}")
        print()
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

    summary = PipelineSummary()

    # ── Run pipeline ────────────────────────────────────────────
    # run_pipeline is a generator — frames are never stored in bulk
    for frame_output in run_pipeline(
        frames_folder=FRAMES_FOLDER,
        preprocess_config=PREPROCESS_CONFIG,
        detector_config=CUSTOM_DETECTOR_CONFIG,
        tracker_config=CUSTOM_TRACKER_CONFIG,
        lane_config=CUSTOM_LANE_CONFIG,
    ):
        # Update running summary
        summary.update(frame_output)

        # 🔽 ADD HERE
        debug_frame = frame_output.get("debug_frame")

        if debug_frame is not None:
            debug_frame_bgr = cv2.cvtColor(debug_frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("Traffic View", debug_frame_bgr)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # ── Per-frame output ────────────────────────────────────
            fid          = frame_output["frame_id"]
            lane_counts  = frame_output["lane_counts"]
            vehicles     = frame_output["vehicles"]
            emerg_lane   = frame_output["emergency_lane"]

            # Print concise one-line summary per frame
            veh_count = len(vehicles)
            emerg_str = f"  ⚠ EMERGENCY → {emerg_lane}" if emerg_lane else ""
            print(f"Frame {fid:>5} | Vehicles: {veh_count:>3} | Lanes: {lane_counts}{emerg_str}")

            # Optionally print full vehicle details (comment out for cleaner output)
            if vehicles:
                for v in vehicles:
                    print(
                        f"         ID:{v['id']:>3}  {v['class']:<12} "
                        f"conf:{v['confidence']:.2f}  lane:{v['lane']}  "
                        f"dir:{v['direction']}  bbox:{v['bbox']}"
                    )

            # Update running summary
            summary.update(frame_output)

            # ── Final summary ────────────────────────────────────────────
            summary.print_summary()


# ─────────────────────────────────────────────
#  STANDALONE DEMO (no real frames needed)
# ─────────────────────────────────────────────
# def demo_with_mock_output():
#     """
#     Shows what the per-frame output structure looks like
#     without needing a real frames folder or YOLO model.
#     """
#     sample_output = {
#         "frame_id": 12,
#         "lane_counts": {
#             "lane_1": 1,
#             "lane_2": 2,
#             "lane_3": 0,
#             "lane_4": 1,
#         },
#         "vehicles": [
#             {
#                 "id": 0,
#                 "lane": "lane_1",
#                 "bbox": [10, 200, 150, 350],
#                 "class": "car",
#                 "confidence": 0.87,
#                 "direction": "down",
#             },
#             {
#                 "id": 1,
#                 "lane": "lane_2",
#                 "bbox": [170, 180, 310, 400],
#                 "class": "bus",
#                 "confidence": 0.91,
#                 "direction": "stationary",
#             },
#             {
#                 "id": 2,
#                 "lane": "lane_2",
#                 "bbox": [200, 220, 290, 360],
#                 "class": "motorcycle",
#                 "confidence": 0.65,
#                 "direction": "down",
#             },
#             {
#                 "id": 3,
#                 "lane": "lane_4",
#                 "bbox": [490, 150, 630, 420],
#                 "class": "truck",
#                 "confidence": 0.78,
#                 "direction": "up",
#             },
#         ],
#         "emergency_lane": None,
#     }

#     print("\n── Sample Frame Output Structure ──────────────────")
#     print(json.dumps(sample_output, indent=2))
#     print("───────────────────────────────────────────────────\n")


if __name__ == "__main__":
    # Show the expected output structure first
    #demo_with_mock_output()

    # Then run the real pipeline (requires FRAMES_FOLDER to exist)
    main()
