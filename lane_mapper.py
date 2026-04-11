"""
lane_mapper.py
--------------
Maps detected vehicle bounding boxes to configurable lane polygons.

Responsibilities:
  - Define lane boundaries as polygons
  - Assign each vehicle to the correct lane using bounding box center
  - Count vehicles per lane
  - Flag lanes containing emergency vehicles
  - Return IDs of specific vehicles that triggered emergency flag

Fix applied (Bug 2):
  - Removed first-frame area-only emergency trigger in detect_emergency_lanes().
    Previously, any truck/bus appearing for the first time (prev_centroid=None)
    with a bounding box area >= 18000px^2 was immediately flagged as an emergency
    vehicle with zero motion context. At 640x480 a moderately sized truck easily
    clears 18000px^2, so every new large vehicle entering the frame triggered a
    false emergency. Now the heuristic is skipped when prev_centroid is None.
    The vehicle is re-evaluated on its second frame when real motion data exists.

Fix applied (Bug 3):
  - assign_lane() no longer force-assigns out-of-polygon detections via the
    x-position fallback. The fallback bucketed every centroid that missed all
    lane polygons into an arbitrary lane based purely on its x coordinate —
    meaning partial detections at frame edges, mis-sized bounding boxes, and
    EmergencyLightDetector blobs that landed outside all polygons all got
    silently assigned a lane and counted/flagged. assign_lane() now returns
    None for out-of-polygon centroids.
  - count_vehicles_per_lane() now reads lane with no default (not "unknown"),
    so None lanes fall into the "unknown" bucket explicitly rather than
    matching a real lane name by accident.
  - detect_emergency_lanes() now skips vehicles whose lane is None — an
    out-of-polygon vehicle cannot meaningfully trigger a lane emergency alert.
"""

import cv2
import numpy as np

# Coordinates are for resized frame 640x480
LANE_CONFIG = {
    "left_road": [
        (0, 250), (200, 200), (200, 480), (0, 480)
    ],
    "bottom_road": [
        (200, 300), (450, 300), (450, 480), (200, 480)
    ],
    "right_road": [
        (450, 250), (640, 200), (640, 480), (450, 480)
    ],
    "top_road": [
        (200, 0), (450, 0), (450, 250), (200, 250)
    ],
}

EMERGENCY_CLASSES = {"ambulance", "fire truck", "firetruck", "fire_truck"}
EMERGENCY_BBOX_AREA_THRESHOLD = 18000
EMERGENCY_SPEED_THRESHOLD = 10


class LaneMapper:
    def __init__(self, lane_config: dict = None):
        self.lanes = lane_config or LANE_CONFIG

    def assign_lane(self, bbox: list):
        """
        Returns the lane name string if the bbox centroid falls inside a
        defined polygon, or None if it falls outside all polygons.

        FIX (Bug 3): previously fell back to x-position bucketing when no
        polygon matched, force-assigning every out-of-polygon centroid to
        some lane. This caused edge-clipped detections and EmergencyLight
        blobs outside all polygons to pollute lane counts and emergency flags.
        Now returns None so callers can explicitly ignore unassigned detections.
        """
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        point = (cx, cy)

        for lane_name, polygon in self.lanes.items():
            poly_np = np.array(polygon, dtype=np.int32)
            inside = cv2.pointPolygonTest(poly_np, point, False)
            if inside >= 0:
                return lane_name

        # Centroid is outside every defined polygon — do not force-assign.
        return None

    def count_vehicles_per_lane(self, vehicle_list: list) -> dict:
        counts = {lane: 0 for lane in self.lanes}
        for vehicle in vehicle_list:
            # FIX (Bug 3): no default — None (out-of-polygon) must not
            # accidentally match a real lane name string.
            lane = vehicle.get("lane")
            if lane in counts:
                counts[lane] += 1
            else:
                # None lanes and any other unrecognised value go here.
                counts["unknown"] = counts.get("unknown", 0) + 1
        return counts

    def detect_emergency_lanes(self, vehicle_list: list) -> tuple:
        """
        Returns (emergency_lanes, emergency_vehicle_ids).

        emergency_lanes:       list of lane name strings
        emergency_vehicle_ids: set of vehicle IDs that directly
                               triggered the emergency flag — NOT all
                               vehicles in those lanes, only the specific
                               truck/bus that passed the heuristic check.
        """
        emergency_lanes       = []
        emergency_vehicle_ids = set()

        for vehicle in vehicle_list:
            cls  = vehicle.get("class", "").lower()
            bbox = vehicle.get("bbox", [0, 0, 0, 0])
            # FIX (Bug 3): no default — keep None as-is so the guard below
            # can skip out-of-polygon vehicles cleanly.
            lane = vehicle.get("lane")
            vid  = vehicle.get("id")

            # FIX (Bug 3): skip vehicles with no valid lane assignment.
            # An out-of-polygon vehicle cannot meaningfully flag a lane.
            if lane is None:
                continue

            # Class-name check — works once model is fine-tuned on ambulance
            if cls in EMERGENCY_CLASSES:
                if lane not in emergency_lanes:
                    emergency_lanes.append(lane)
                if vid is not None:
                    emergency_vehicle_ids.add(vid)
                continue

            if cls in {"truck", "bus"}:
                prev = vehicle.get("prev_centroid")

                # FIX (Bug 2): never flag on first appearance (prev_centroid=None).
                # The original code triggered on area alone when prev was None,
                # meaning every new large truck/bus entering the frame was
                # immediately classified as an emergency vehicle before any
                # motion data existed. Now we simply wait for the second frame
                # when both area AND speed can be evaluated together.
                if prev is None:
                    continue

                # Has motion history — use full area + speed check
                if self._is_emergency_heuristic(bbox, vehicle):
                    if lane not in emergency_lanes:
                        emergency_lanes.append(lane)
                    if vid is not None:
                        emergency_vehicle_ids.add(vid)

        return emergency_lanes, emergency_vehicle_ids

    def analyse(self, vehicle_list: list) -> tuple:
        """
        Returns (lane_counts, emergency_lanes, emergency_vehicle_ids).
        Previously returned (lane_counts, emergency_lanes) — now includes
        emergency_vehicle_ids so pipeline can track the specific vehicles.
        """
        lane_counts = self.count_vehicles_per_lane(vehicle_list)
        emergency_lanes, emergency_vehicle_ids = self.detect_emergency_lanes(vehicle_list)
        return lane_counts, emergency_lanes, emergency_vehicle_ids

    def get_lane_boundaries(self) -> dict:
        return dict(self.lanes)

    @staticmethod
    def _bbox_area(bbox: list) -> int:
        x1, y1, x2, y2 = bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    @staticmethod
    def _centroid_speed(vehicle: dict) -> float:
        curr = vehicle.get("centroid")
        prev = vehicle.get("prev_centroid")
        if curr is None or prev is None:
            return 0.0
        dx = curr[0] - prev[0]
        dy = curr[1] - prev[1]
        return (dx**2 + dy**2) ** 0.5

    def _is_emergency_heuristic(self, bbox: list, vehicle: dict) -> bool:
        area  = self._bbox_area(bbox)
        speed = self._centroid_speed(vehicle)
        return area >= EMERGENCY_BBOX_AREA_THRESHOLD and speed >= EMERGENCY_SPEED_THRESHOLD