"""
lane_mapper.py
--------------
Maps detected vehicle bounding boxes to configurable lane polygons.

Responsibilities:
  - Define lane boundaries as polygons
  - Assign each vehicle to the correct lane using bounding box center
  - Count vehicles per lane
  - Flag lanes containing emergency vehicles
"""

import cv2
import numpy as np

# Coordinates are for resized frame 640x480
LANE_CONFIG = {
    "lane_1": [(0, 220), (220, 220), (220, 480), (0, 480)],
    "lane_2": [(220, 180), (380, 180), (380, 480), (220, 480)],
    "lane_3": [(380, 140), (540, 140), (540, 480), (380, 480)],
    "lane_4": [(540, 140), (640, 140), (640, 480), (540, 480)],
}

EMERGENCY_CLASSES = {"ambulance", "fire truck", "firetruck", "fire_truck"}
EMERGENCY_BBOX_AREA_THRESHOLD = 18000
EMERGENCY_SPEED_THRESHOLD = 25


class LaneMapper:
    def __init__(self, lane_config: dict = None):
        self.lanes = lane_config or LANE_CONFIG

    def assign_lane(self, bbox: list) -> str:
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        point = (cx, cy)

        for lane_name, polygon in self.lanes.items():
            poly_np = np.array(polygon, dtype=np.int32)
            inside = cv2.pointPolygonTest(poly_np, point, False)
            if inside >= 0:
                return lane_name

        return "unknown"

    def count_vehicles_per_lane(self, vehicle_list: list) -> dict:
        counts = {lane: 0 for lane in self.lanes}
        counts["unknown"] = 0

        for vehicle in vehicle_list:
            lane = vehicle.get("lane", "unknown")
            if lane in counts:
                counts[lane] += 1
            else:
                counts["unknown"] += 1

        return counts

    def detect_emergency_lane(self, vehicle_list: list) -> str | None:
        for vehicle in vehicle_list:
            cls = vehicle.get("class", "").lower()
            bbox = vehicle.get("bbox", [0, 0, 0, 0])
            lane = vehicle.get("lane", "unknown")

            if cls in EMERGENCY_CLASSES:
                return lane

            if self._is_emergency_heuristic(bbox, vehicle):
                return lane

        return None

    def analyse(self, vehicle_list: list) -> tuple:
        lane_counts = self.count_vehicles_per_lane(vehicle_list)
        emergency_lane = self.detect_emergency_lane(vehicle_list)
        return lane_counts, emergency_lane

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
        area = self._bbox_area(bbox)
        speed = self._centroid_speed(vehicle)
        return area >= EMERGENCY_BBOX_AREA_THRESHOLD and speed >= EMERGENCY_SPEED_THRESHOLD