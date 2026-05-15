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

# Coordinates are for resized frame 1280x720
LANE_CONFIG = {
    "left_road": [(8,399),(397,256),(492,677),(7,692)],
    "bottom_road": [(1083,411),(1269,635),(1277,714),(801,717)],
    "right_road": [(1005,81),(1069,134),(1028,304),(784,164)],
    "top_road": [(405,51),(466,40),(723,159),(455,209)]
}

# Optional global shift if camera moved
GLOBAL_SHIFT_X = 0
GLOBAL_SHIFT_Y = 0

if GLOBAL_SHIFT_X != 0 or GLOBAL_SHIFT_Y != 0:
    for lane in LANE_CONFIG:
        LANE_CONFIG[lane] = [(x + GLOBAL_SHIFT_X, y + GLOBAL_SHIFT_Y) for (x, y) in LANE_CONFIG[lane]]

EMERGENCY_CLASSES = {"ambulance", "fire truck", "firetruck", "fire_truck"}
EMERGENCY_BBOX_AREA_THRESHOLD = 15000
EMERGENCY_SPEED_THRESHOLD = 15
MIN_LANE_VEHICLE_AREA = 1000



class LaneMapper:
    def __init__(self, lane_config: dict = None):
        self.lanes = lane_config or LANE_CONFIG

    def assign_lane(self, bbox: list):
        """
        Returns the lane name string if the bbox centroid falls inside a
        defined polygon. If outside all polygons, falls back to the
        nearest polygon within a tolerance (vehicles near the stop line
        at the intersection center).

        Returns None only if the vehicle is far from all polygons.
        """
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        point = (cx, cy)

        # First pass: check if centroid is inside any polygon
        for lane_name, polygon in self.lanes.items():
            poly_np = np.array(polygon, dtype=np.int32)
            inside = cv2.pointPolygonTest(poly_np, point, False)
            if inside >= 0:
                return lane_name

        # Second pass: find nearest polygon (for vehicles near the stop line)
        # pointPolygonTest with measureDist=True returns negative distance
        # for points outside (closer to 0 = closer to the edge)
        best_lane = None
        best_dist = float("inf")
        MAX_FALLBACK_DIST = 150  # pixels — don't assign if too far

        for lane_name, polygon in self.lanes.items():
            poly_np = np.array(polygon, dtype=np.int32)
            dist = cv2.pointPolygonTest(poly_np, point, True)
            # dist is negative for outside points; abs(dist) = distance to edge
            abs_dist = abs(dist)
            if abs_dist < best_dist:
                best_dist = abs_dist
                best_lane = lane_name

        if best_dist <= MAX_FALLBACK_DIST:
            return best_lane

        return None

    def count_vehicles_per_lane(self, vehicle_list: list) -> dict:
        counts = {lane: 0 for lane in self.lanes}
        for vehicle in vehicle_list:
            bbox = vehicle.get("bbox")
            if bbox and self._bbox_area(bbox) < MIN_LANE_VEHICLE_AREA:
                continue

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