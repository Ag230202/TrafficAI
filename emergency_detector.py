"""
emergency_detector.py
---------------------
Detects emergency vehicles using flashing light signatures —
works independently of YOLO class labels.

Strategy:
  Emergency vehicles (fire trucks, ambulances, police) emit bright
  coloured flashes (red, blue, amber) that saturate specific HSV
  colour ranges. This module:
    1. Detects large bright blobs in red/blue/amber HSV ranges
    2. Checks if the blob is large enough to be a vehicle (not a traffic light)
    3. Checks that enough colour pixels exist inside the blob (not just area)
    4. Maps the blob centroid to the nearest lane using LaneMapper
    5. Returns a list of lane names where emergency lights are detected

This runs IN ADDITION to the YOLO-based heuristic — not instead of it.
Either method triggering is enough to flag a lane as emergency.

Fixes applied (Bug 1):
  - Raised S and V minimums from 150 → 180/200 to stop CLAHE-boosted
    brake lights, traffic signals and sunlit vehicles from triggering.
  - Raised min_blob_area from 400 → 800 to filter small reflections.
  - Raised min_color_pixels from 150 → 500 to require a dense colour
    cluster, not just a sparse scatter of matching pixels.
  - Actually ENFORCE min_color_pixels in the contour loop — the original
    code defined it in config but never checked it.
"""

import cv2
import numpy as np


# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
EMERGENCY_LIGHT_CONFIG = {
    # Minimum blob area in pixels to be considered a vehicle light source
    # (not a small traffic light or reflection)
    # FIX: raised from 400 → 800
    "min_blob_area": 800,

    # Maximum blob area — very large blobs are likely ROI artefacts
    "max_blob_area": 80000,

    # HSV colour ranges for emergency light colours
    # OpenCV HSV: H=0-179, S=0-255, V=0-255
    #
    # FIX: S minimum raised 150 → 180, V minimum raised 150 → 200.
    # Rationale: genuine emergency strobes are extremely saturated and
    # bright. CLAHE contrast enhancement (used in preprocessing) pushes
    # ordinary brake lights and traffic signals into the old S≥150/V≥150
    # bands, causing false positives on every frame. S≥180/V≥200 targets
    # only the intense, near-white-hot flashes of actual emergency lights.
    "color_ranges": {
        # Red — fire trucks, ambulances, police (red wraps around hue)
        "red_low":  {"lower": (0,   180, 200), "upper": (10,  255, 255)},
        "red_high": {"lower": (165, 180, 200), "upper": (179, 255, 255)},
        # Blue — police lights
        "blue":     {"lower": (100, 180, 200), "upper": (130, 255, 255)},
        # Amber/orange — ambulance, roadwork, some fire trucks
        "amber":    {"lower": (10,  180, 200), "upper": (25,  255, 255)},
    },

    # Minimum number of active colour pixels INSIDE a blob's bounding box.
    # FIX: raised from 150 → 500, and is now actually enforced in detect().
    # This rejects blobs whose bounding box area passes the size check but
    # whose interior is mostly dark — e.g. a large vehicle silhouette with
    # only a few bright pixels on its edge.
    "min_color_pixels": 500,

    # Whether to draw debug blobs on the debug frame
    "draw_debug": True,
}


class EmergencyLightDetector:
    """
    Detects emergency vehicle lights by colour blob analysis.

    Works on BGR frames (before BGR→RGB conversion).
    Call detect() once per frame alongside YOLO detection.
    """

    def __init__(self, config: dict = None):
        self.cfg = config or EMERGENCY_LIGHT_CONFIG

    def detect(self, frame_bgr: np.ndarray, lane_mapper, debug_frame=None) -> list:
        """
        Detects emergency light blobs in the frame and maps them to lanes.

        frame_bgr:   BGR np.ndarray (before colour conversion)
        lane_mapper: LaneMapper instance — used to assign blobs to lanes
        debug_frame: optional RGB frame to draw blobs on for visualisation

        Returns: list of lane name strings where emergency lights found
        """
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        emergency_lanes = []
        cfg = self.cfg

        # Build combined mask for all emergency light colours
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

        color_ranges = cfg["color_ranges"]
        for color_name, ranges in color_ranges.items():
            if "low" in color_name and color_name.replace("low", "high") in color_ranges:
                # Red wraps around hue — OR both ranges together
                mask_low  = cv2.inRange(hsv,
                    np.array(color_ranges["red_low"]["lower"]),
                    np.array(color_ranges["red_low"]["upper"]))
                mask_high = cv2.inRange(hsv,
                    np.array(color_ranges["red_high"]["lower"]),
                    np.array(color_ranges["red_high"]["upper"]))
                combined_mask = cv2.bitwise_or(combined_mask, mask_low)
                combined_mask = cv2.bitwise_or(combined_mask, mask_high)
            elif "high" in color_name:
                continue  # already handled above with red_low
            else:
                mask = cv2.inRange(hsv,
                    np.array(ranges["lower"]),
                    np.array(ranges["upper"]))
                combined_mask = cv2.bitwise_or(combined_mask, mask)

        # Morphological close to join nearby bright pixels into blobs
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN,
                                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

        # Find contours of bright blobs
        contours, _ = cv2.findContours(
            combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < cfg["min_blob_area"] or area > cfg["max_blob_area"]:
                continue

            # Get bounding box and centroid
            x, y, w, h = cv2.boundingRect(cnt)

            # FIX: enforce min_color_pixels — count how many qualifying
            # colour pixels actually exist inside this blob's bounding box.
            # The original code defined min_color_pixels in config but
            # never checked it, so large low-density blobs (brake lights,
            # sunlit bonnets) passed straight through.
            color_pixel_count = cv2.countNonZero(combined_mask[y:y + h, x:x + w])
            if color_pixel_count < cfg["min_color_pixels"]:
                continue

            # Assign to lane using bounding box centroid
            bbox = [x, y, x + w, y + h]
            lane = lane_mapper.assign_lane(bbox)

            if lane and lane not in emergency_lanes:
                emergency_lanes.append(lane)

            # Draw on debug frame if provided
            if debug_frame is not None and cfg.get("draw_debug", True):
                cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (255, 165, 0), 2)
                cv2.putText(
                    debug_frame,
                    f"LIGHT {lane}",
                    (x, max(15, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 165, 0),
                    1,
                )

        return emergency_lanes