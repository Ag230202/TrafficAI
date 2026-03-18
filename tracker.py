"""
tracker.py
----------
Lightweight centroid-based vehicle tracker.

Assigns persistent IDs to detected vehicles across frames by matching
new detections to existing tracks using nearest-centroid distance.
No heavy dependencies — uses only NumPy.

Designed for low RAM usage: only minimal history is stored per track.
"""

import numpy as np


# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
TRACKER_CONFIG = {
    # Maximum pixel distance to match a detection to an existing track
    "max_distance": 80,

    # Number of consecutive frames a track can be "missing" before removal
    "max_lost_frames": 10,

    # Minimum number of frames a track must be seen to be considered stable
    "min_hits": 2,
}


# ─────────────────────────────────────────────
#  TRACK: Represents a single tracked vehicle
# ─────────────────────────────────────────────
class Track:
    """
    Stores state for a single tracked vehicle.

    Keeps only the last two centroids to compute direction —
    avoids accumulating a long history list in memory.
    """

    def __init__(self, track_id: int, centroid: tuple, bbox: list, cls: str, conf: float):
        self.track_id    = track_id         # Unique vehicle ID
        self.centroid    = centroid         # Current (cx, cy)
        self.prev_centroid = None           # Previous (cx, cy) — for direction
        self.bbox        = bbox             # [x1, y1, x2, y2]
        self.cls         = cls              # Class label string
        self.conf        = conf             # Detection confidence
        self.hits        = 1               # Frames this track has been seen
        self.lost_frames = 0               # Consecutive frames without match
        self.direction   = "unknown"        # Movement direction string

    def update(self, centroid: tuple, bbox: list, cls: str, conf: float):
        """Update track with a new matched detection."""
        self.prev_centroid = self.centroid  # Shift current → previous
        self.centroid      = centroid
        self.bbox          = bbox
        self.cls           = cls
        self.conf          = conf
        self.hits         += 1
        self.lost_frames   = 0             # Reset loss counter on match
        self.direction     = self._compute_direction()

    def mark_lost(self):
        """Called when no detection matched this track in a frame."""
        self.lost_frames += 1

    def is_active(self, max_lost: int) -> bool:
        """Returns True if the track should still be retained."""
        return self.lost_frames <= max_lost

    def _compute_direction(self) -> str:
        """
        Computes movement direction from previous to current centroid.
        Uses dominant axis (X vs Y) and sign to determine direction.
        Returns: "up", "down", "left", "right", or "stationary".
        """
        if self.prev_centroid is None:
            return "unknown"

        dx = self.centroid[0] - self.prev_centroid[0]
        dy = self.centroid[1] - self.prev_centroid[1]

        # Threshold: ignore very small jitter
        if abs(dx) < 3 and abs(dy) < 3:
            return "stationary"

        # Dominant axis determines direction
        if abs(dx) >= abs(dy):
            return "right" if dx > 0 else "left"
        else:
            return "down" if dy > 0 else "up"


# ─────────────────────────────────────────────
#  CENTROID TRACKER
# ─────────────────────────────────────────────
class CentroidTracker:
    """
    Assigns and maintains vehicle IDs across frames using centroid matching.

    Algorithm per frame:
      1. Compute centroids for all new detections.
      2. Build a distance matrix between existing track centroids and new ones.
      3. Greedily match closest pairs within max_distance threshold.
      4. Unmatched existing tracks → mark as lost (increment lost_frames).
      5. Unmatched new detections → create new Track.
      6. Remove tracks that have been lost too long.
    """

    def __init__(self, config: dict = None):
        cfg = config or TRACKER_CONFIG
        self.max_distance  = cfg.get("max_distance", 80)
        self.max_lost      = cfg.get("max_lost_frames", 10)
        self.min_hits      = cfg.get("min_hits", 2)

        self.tracks: dict[int, Track] = {}  # track_id → Track
        self._next_id = 0                   # Auto-incrementing ID counter

    # ── Public API ───────────────────────────
    def update(self, detections: list) -> list:
        """
        Process detections for the current frame.

        detections: list of dicts with keys:
            bbox  → [x1, y1, x2, y2]
            class → str
            confidence → float

        Returns: list of active Track objects (hits >= min_hits).
        """
        if not detections:
            # No detections — mark all tracks as lost
            for track in self.tracks.values():
                track.mark_lost()
            self._remove_stale_tracks()
            return self._get_active_tracks()

        # Step 1: Compute new centroids
        new_centroids = [self._bbox_to_centroid(d["bbox"]) for d in detections]

        if not self.tracks:
            # No existing tracks — register all detections as new
            for centroid, det in zip(new_centroids, detections):
                self._register(centroid, det)
            return self._get_active_tracks()

        # Step 2: Build distance matrix [existing_tracks × new_detections]
        track_ids   = list(self.tracks.keys())
        old_cents   = [self.tracks[tid].centroid for tid in track_ids]
        dist_matrix = self._compute_distances(old_cents, new_centroids)

        # Step 3: Greedy matching — smallest distance first
        matched_track_indices = set()
        matched_det_indices   = set()

        # Get sorted indices by distance (row, col)
        sorted_idx = np.dstack(
            np.unravel_index(np.argsort(dist_matrix, axis=None), dist_matrix.shape)
        )[0]

        for row, col in sorted_idx:
            if row in matched_track_indices or col in matched_det_indices:
                continue
            if dist_matrix[row, col] > self.max_distance:
                break  # All remaining pairs exceed threshold

            tid = track_ids[row]
            det = detections[col]
            self.tracks[tid].update(
                centroid=new_centroids[col],
                bbox=det["bbox"],
                cls=det["class"],
                conf=det["confidence"],
            )
            matched_track_indices.add(row)
            matched_det_indices.add(col)

        # Step 4: Unmatched existing tracks → mark lost
        for row, tid in enumerate(track_ids):
            if row not in matched_track_indices:
                self.tracks[tid].mark_lost()

        # Step 5: Unmatched new detections → register as new tracks
        for col, (centroid, det) in enumerate(zip(new_centroids, detections)):
            if col not in matched_det_indices:
                self._register(centroid, det)

        # Step 6: Remove stale tracks
        self._remove_stale_tracks()

        return self._get_active_tracks()

    def reset(self):
        """Clears all tracks. Useful between video sequences."""
        self.tracks.clear()
        self._next_id = 0

    # ── Internal Helpers ─────────────────────
    def _register(self, centroid: tuple, det: dict):
        """Creates a new Track from a detection."""
        track = Track(
            track_id=self._next_id,
            centroid=centroid,
            bbox=det["bbox"],
            cls=det["class"],
            conf=det["confidence"],
        )
        self.tracks[self._next_id] = track
        self._next_id += 1

    def _remove_stale_tracks(self):
        """Removes tracks that have been lost for too many frames."""
        stale = [tid for tid, t in self.tracks.items() if not t.is_active(self.max_lost)]
        for tid in stale:
            del self.tracks[tid]

    def _get_active_tracks(self) -> list:
        """Returns tracks that have been seen enough times to be reliable."""
        return [t for t in self.tracks.values() if t.hits >= self.min_hits]

    @staticmethod
    def _bbox_to_centroid(bbox: list) -> tuple:
        """Computes (cx, cy) from [x1, y1, x2, y2]."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @staticmethod
    def _compute_distances(centroids_a: list, centroids_b: list) -> np.ndarray:
        """
        Computes Euclidean distance matrix between two lists of (cx, cy) tuples.
        Returns shape: (len(a), len(b))
        """
        a = np.array(centroids_a, dtype=np.float32)  # (N, 2)
        b = np.array(centroids_b, dtype=np.float32)  # (M, 2)

        # Broadcasting: (N, 1, 2) - (1, M, 2) → (N, M, 2)
        diff = a[:, np.newaxis, :] - b[np.newaxis, :, :]
        return np.sqrt((diff ** 2).sum(axis=2))       # (N, M)
