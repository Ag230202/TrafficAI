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
    "min_blob_area": 1000,

    # Maximum blob area — very large blobs are likely ROI artefacts
    "max_blob_area": 80000,

    # HSV colour ranges for emergency light colours
    # OpenCV HSV: H=0-179, S=0-255, V=0-255
    # Adjusted S/V limits to 150 to capture daytime paint markings on static ambulances.
    "color_ranges": {
        # Red — fire trucks, ambulances, police. V raised to 240 to ignore red paint.
        "red_low":  {"lower": (0,   200, 240), "upper": (10,  255, 255)},
        "red_high": {"lower": (165, 200, 240), "upper": (179, 255, 255)},
        # Blue — police lights
        "blue":     {"lower": (100, 200, 200), "upper": (130, 255, 255)},
        # Amber/orange — ambulance, roadwork, some fire trucks
        "amber":    {"lower": (10,  200, 200), "upper": (25,  255, 255)},
    },

    # Minimum number of active colour pixels INSIDE a blob's bounding box.
    "min_color_pixels": 800,

    # Whether to draw debug blobs on the debug frame
    "draw_debug": False,
}


class EmergencyLightDetector:
    """
    Detects emergency vehicle lights by colour blob analysis.

    Works on BGR frames (before BGR→RGB conversion).
    Call detect() once per frame alongside YOLO detection.
    """

    def __init__(self, config: dict = None):
        self.cfg = config or EMERGENCY_LIGHT_CONFIG

    def detect(self, frame_bgr: np.ndarray, lane_mapper, debug_frame=None, vehicles=None) -> dict:
        """
        Detects emergency light blobs in the frame and maps them to lanes. Also
        performs high-precision color density checks within vehicle bounding boxes
        to identify unique emergency vehicles.

        frame_bgr:   BGR np.ndarray (before colour conversion)
        lane_mapper: LaneMapper instance — used to assign blobs to lanes
        debug_frame: optional RGB frame to draw blobs on for visualisation
        vehicles:    optional list of tracked vehicles to check for internal lights

        Returns: dict with keys "detected_blobs" (list of dicts) and "matched_vehicle_ids" (set)
        """
        # FIX: No vehicles tracked → no emergency vehicle possible.
        # Skip all expensive color analysis to avoid false positives from
        # traffic lights, building reflections, or sunlit surfaces.
        if not vehicles:
            return {"detected_blobs": [], "matched_vehicle_ids": set()}

        # Build a set of lanes that currently have at least one tracked vehicle.
        # Color blobs will only be accepted if a real vehicle occupies the same lane.
        occupied_lanes = set(
            v.get("lane") for v in vehicles if v.get("lane") is not None
        )
        if not occupied_lanes:
            return {"detected_blobs": [], "matched_vehicle_ids": set()}

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        detected_blobs = []
        matched_vehicle_ids = set()
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

        # High-precision color density check within each vehicle bounding box.
        # FIX: Only process vehicles that have a valid lane assignment —
        # an out-of-polygon vehicle cannot meaningfully flag a lane.
        h_img, w_img = combined_mask.shape[:2]
        for v in vehicles:
            if v.get("lane") is None:
                continue  # FIX: skip vehicles with no valid lane
            bbox = v.get("bbox")
            if bbox:
                vx1, vy1, vx2, vy2 = bbox
                x1_c = max(0, min(vx1, w_img - 1))
                y1_c = max(0, min(vy1, h_img - 1))
                x2_c = max(0, min(vx2, w_img - 1))
                y2_c = max(0, min(vy2, h_img - 1))

                if x2_c > x1_c and y2_c > y1_c:
                    crop = combined_mask[y1_c:y2_c, x1_c:x2_c]
                    color_pixels = cv2.countNonZero(crop)
                    # Require a dense cluster of active emergency color pixels (at least 500px)
                    # inside the vehicle bounding box to classify as emergency vehicle.
                    if color_pixels >= 500:
                        vid = v.get("id")
                        if vid is not None and vid != -1:
                            matched_vehicle_ids.add(vid)

        # Morphological close to join nearby bright pixels into blobs for lane mapping
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        closed_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        closed_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN,
                                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

        # Find contours of bright blobs
        contours, _ = cv2.findContours(
            closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < cfg["min_blob_area"] or area > cfg["max_blob_area"]:
                continue

            # Get bounding box and centroid
            x, y, w, h = cv2.boundingRect(cnt)

            # Enforce min_color_pixels
            color_pixel_count = cv2.countNonZero(closed_mask[y:y + h, x:x + w])
            if color_pixel_count < cfg["min_color_pixels"]:
                continue

            # Assign to lane using bounding box centroid
            bbox = [x, y, x + w, y + h]
            lane = lane_mapper.assign_lane(bbox)

            # FIX: Only flag a lane if a real tracked vehicle is currently in it.
            # This prevents traffic lights, building reflections, and roadway
            # markings from triggering an emergency alert when no vehicle exists.
            if lane and lane in occupied_lanes:
                detected_blobs.append({
                    "lane": lane,
                    "bbox": bbox
                })

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

        return {
            "detected_blobs": detected_blobs,
            "matched_vehicle_ids": matched_vehicle_ids
        }