"""
detector.py
-----------
YOLOv8n vehicle detection module.

Responsibilities:
  - Load YOLOv8n model once and reuse across frames
  - Run inference on preprocessed frames
  - Filter detections by class and confidence
  - Return structured detection results per frame
"""

import numpy as np

# ─────────────────────────────────────────────
#  DETECTION CONFIGURATION
# ─────────────────────────────────────────────
DETECTOR_CONFIG = {
    # YOLOv8 model variant — "yolov8n.pt" is smallest (best for CPU)
    "model_path": "yolov8n.pt",

    # Minimum confidence to keep a detection
    "confidence_threshold": 0.4,

    # COCO class IDs to detect (vehicles only)
    # 2=car, 3=motorcycle, 5=bus, 7=truck
    # YOLOv8n does not include ambulance/fire truck in COCO,
    # so those are handled separately via heuristics in lane_mapper.py
    "target_class_ids": {2, 3, 5, 7},

    # Human-readable labels for each COCO class ID we care about
    "class_labels": {
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    },

    # Inference image size (must match preprocessing resize)
    "imgsz": 640,

    # Force CPU inference — set device="cpu" always
    "device": "cpu",

    # Suppress YOLO console output during inference
    "verbose": False,
}


class VehicleDetector:
    """
    Wraps YOLOv8n for vehicle-only detection.

    Usage:
        detector = VehicleDetector()
        detections = detector.detect(frame, frame_index)
    """

    def __init__(self, config: dict = None):
        cfg = config or DETECTOR_CONFIG
        self.conf_threshold   = cfg.get("confidence_threshold", 0.4)
        self.target_class_ids = cfg.get("target_class_ids", {2, 3, 5, 7})
        self.class_labels     = cfg.get("class_labels", {})
        self.imgsz            = cfg.get("imgsz", 640)
        self.device           = cfg.get("device", "cpu")
        self.verbose          = cfg.get("verbose", False)

        # Load model once — kept in memory across all frames
        self.model = self._load_model(cfg.get("model_path", "yolov8n.pt"))

    # ─────────────────────────────────────────
    #  MODEL LOADING
    # ─────────────────────────────────────────
    def _load_model(self, model_path: str):
        """
        Loads YOLOv8 model from file or downloads if not present.
        Model is loaded once and reused — no repeated disk I/O.
        """
        try:
            from ultralytics import YOLO
            print(f"[Detector] Loading model: {model_path} on {self.device}")
            model = YOLO(model_path)
            model.to(self.device)
            print("[Detector] Model loaded successfully.")
            return model
        except ImportError:
            raise ImportError(
                "ultralytics package not found. "
                "Install it with: pip install ultralytics"
            )
        except Exception as e:
            raise RuntimeError(f"[Detector] Failed to load model '{model_path}': {e}")

    # ─────────────────────────────────────────
    #  INFERENCE
    # ─────────────────────────────────────────
    def detect(self, frame: np.ndarray, frame_index: int = 0) -> list:
        """
        Runs YOLOv8 inference on a single preprocessed frame.

        frame: RGB np.ndarray (H, W, 3) — output from preprocessing pipeline
        frame_index: int — used for logging/debugging only

        Returns: list of detection dicts:
            {
                "bbox":       [x1, y1, x2, y2],
                "class":      str,
                "class_id":   int,
                "confidence": float
            }
        """
        # Run inference — single frame, no batching
        results = self.model.predict(
            source=frame,
            imgsz=self.imgsz,
            conf=self.conf_threshold,
            device=self.device,
            verbose=self.verbose,
        )

        # Extract and filter detections
        detections = self._parse_results(results)

        if self.verbose:
            print(f"[Detector] Frame {frame_index}: {len(detections)} vehicle(s) detected.")

        return detections

    # ─────────────────────────────────────────
    #  RESULT PARSING
    # ─────────────────────────────────────────
    def _parse_results(self, results) -> list:
        """
        Parses YOLO result objects into a clean list of detection dicts.

        Filters:
          - Only keeps classes in target_class_ids
          - Confidence already filtered by model.predict(conf=...)

        Returns list of detection dicts.
        """
        detections = []

        # results is a list of Result objects (one per image)
        for result in results:
            boxes = result.boxes  # ultralytics Boxes object

            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                class_id = int(box.cls[0].item())

                # Skip non-vehicle classes
                if class_id not in self.target_class_ids:
                    continue

                confidence = float(box.conf[0].item())

                # Extra confidence guard (model.predict already filters,
                # but keeps this layer explicit for safety)
                if confidence < self.conf_threshold:
                    continue

                # Extract bounding box as integers
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

                label = self.class_labels.get(class_id, f"class_{class_id}")

                detections.append({
                    "bbox":       [x1, y1, x2, y2],
                    "class":      label,
                    "class_id":   class_id,
                    "confidence": round(confidence, 4),
                })

        return detections

    # ─────────────────────────────────────────
    #  UTILITY
    # ─────────────────────────────────────────
    def get_supported_classes(self) -> dict:
        """Returns the class ID → label mapping used by this detector."""
        return dict(self.class_labels)

    def set_confidence_threshold(self, threshold: float):
        """Allows runtime adjustment of the confidence threshold."""
        self.conf_threshold = max(0.0, min(1.0, threshold))
        print(f"[Detector] Confidence threshold updated to: {self.conf_threshold}")
