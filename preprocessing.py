"""
preprocessing.py
----------------
Memory-efficient preprocessing pipeline for traffic video frames.
Designed for YOLO compatibility, CPU-only, low RAM usage (<2GB).
Frames are processed one-by-one using Python generators.
"""

import cv2
import numpy as np
import os

# ─────────────────────────────────────────────
#  GLOBAL PIPELINE CONFIGURATION
#  Edit these values to tune the pipeline.
# ─────────────────────────────────────────────
CONFIG = {
    # Resize dimensions (YOLO-friendly default)
    "resize_width": 640,
    "resize_height": 480,

    # Process only every Nth frame (3 = every 3rd frame)
    "frame_skip": 3,

    # Region of Interest — list of (x1, y1, x2, y2) per lane/ROI
    # Set to None or [] to disable ROI cropping
    # Example: two side-by-side lane ROIs
    "rois": [
        (0, 100, 640, 480),   # Full road region (single ROI example)
    ],

    # Brightness / Contrast  (cv2.convertScaleAbs)
    # alpha: contrast multiplier (1.0 = no change, >1 = more contrast)
    # beta:  brightness offset   (0 = no change, +50 = brighter)
    "alpha":1.2,# 1.2,
    "beta": 15,#15,

    # Gaussian Blur kernel size — must be odd (e.g. 3, 5, 7)
    "blur_kernel":(1,1), #(3, 3),

    # Background subtraction toggle
    "use_background_subtraction": False,

    # Background subtractor parameters
    "bg_history": 500,
    "bg_var_threshold": 16,
}


# ─────────────────────────────────────────────
#  STEP 1 — FRAME LOADER (generator)
# ─────────────────────────────────────────────
def load_frames(folder_path: str):
    """
    Generator that reads frames one-by-one from a folder.
    Yields: (frame_index: int, frame: np.ndarray)

    - Uses sorted file order for consistent sequence.
    - Only loads one frame at a time → minimal RAM usage.
    """
    # Supported image extensions
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")

    # Sort filenames for deterministic ordering
    filenames = sorted(
        f for f in os.listdir(folder_path)
        if f.lower().endswith(valid_ext)
    )

    if not filenames:
        raise FileNotFoundError(f"No image frames found in: {folder_path}")

    for idx, filename in enumerate(filenames):
        filepath = os.path.join(folder_path, filename)
        frame = cv2.imread(filepath)  # Loaded as BGR by default

        if frame is None:
            print(f"[WARNING] Could not read frame: {filepath} — skipping.")
            continue

        yield idx, frame  # Yield one frame, then discard


# ─────────────────────────────────────────────
#  STEP 2 — FRAME SKIPPING
# ─────────────────────────────────────────────
def should_process_frame(frame_index: int, skip: int = 3) -> bool:
    """
    Returns True only for every Nth frame (default: every 3rd).
    Keeps frame_index intact for downstream tracking.
    """
    return frame_index % skip == 0


# ─────────────────────────────────────────────
#  STEP 3 — RESIZE
# ─────────────────────────────────────────────
def resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Resizes frame to the specified (width, height).
    Uses INTER_LINEAR for a balance of speed and quality.
    """
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)


# ─────────────────────────────────────────────
#  STEP 4 — REGION OF INTEREST (ROI)
# ─────────────────────────────────────────────
def apply_roi(frame: np.ndarray, rois: list) -> np.ndarray:
    """
    Masks the frame to keep only the specified ROI region(s).
    Supports multiple ROIs (e.g., multiple lanes).

    rois: list of (x1, y1, x2, y2) tuples
    - If rois is empty/None, returns the original frame unchanged.
    - If multiple ROIs, combines them into a single mask.
    """
    if not rois:
        return frame  # No ROI defined — pass through

    # Create a black mask the same size as the frame
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    for (x1, y1, x2, y2) in rois:
        # Fill each ROI region with white (255) on the mask
        mask[y1:y2, x1:x2] = 255

    # Apply mask: keep only ROI pixels, zero out the rest
    return cv2.bitwise_and(frame, frame, mask=mask)


# ─────────────────────────────────────────────
#  STEP 5 — BRIGHTNESS & CONTRAST ADJUSTMENT
# ─────────────────────────────────────────────
def adjust_brightness_contrast(
    frame: np.ndarray, alpha: float = 1.2, beta: int = 15
) -> np.ndarray:
    """
    Adjusts brightness and contrast of the frame.

    alpha: contrast control (1.0 = unchanged, >1 increases contrast)
    beta:  brightness control (0 = unchanged, positive = brighter)

    Uses cv2.convertScaleAbs which clips values to [0, 255].
    """
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)


# ─────────────────────────────────────────────
#  STEP 6 — NOISE REDUCTION (Gaussian Blur)
# ─────────────────────────────────────────────
def reduce_noise(frame: np.ndarray, kernel_size: tuple = (3, 3)) -> np.ndarray:
    """
    Applies Gaussian Blur to reduce noise.

    kernel_size: must be a tuple of two odd positive integers, e.g. (3,3) or (5,5).
    Larger kernel = more smoothing, but slower.
    """
    return cv2.GaussianBlur(frame, kernel_size, 0)


# ─────────────────────────────────────────────
#  STEP 7 — COLOR CONVERSION (BGR → RGB)
# ─────────────────────────────────────────────
def convert_bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """
    Converts frame from OpenCV BGR format to RGB.
    Required for YOLO models that expect RGB input.
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# ─────────────────────────────────────────────
#  STEP 8 — OPTIONAL: BACKGROUND SUBTRACTION
# ─────────────────────────────────────────────
def create_bg_subtractor(history: int = 500, var_threshold: int = 16):
    """
    Creates and returns a MOG2 background subtractor instance.
    Call this ONCE before the pipeline loop to preserve state.
    """
    return cv2.createBackgroundSubtractorMOG2(
        history=history,
        varThreshold=var_threshold,
        detectShadows=False  # Faster when shadows are ignored
    )


def apply_background_subtraction(
    frame: np.ndarray, bg_subtractor
) -> np.ndarray:
    """
    Applies MOG2 background subtraction to isolate moving objects.

    Returns the foreground mask applied over the original frame.
    bg_subtractor must be the same instance across frames to learn background.
    """
    fg_mask = bg_subtractor.apply(frame)

    # Expand mask to 3 channels so we can bitwise_and with color frame
    fg_mask_3ch = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
    return cv2.bitwise_and(frame, fg_mask_3ch)


# ─────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────
def preprocess_pipeline(folder_path: str, config: dict = None):
    """
    Main generator pipeline. Applies all preprocessing steps sequentially.

    Yields: (frame_index: int, preprocessed_frame: np.ndarray)

    - Reads frames one at a time (memory-safe).
    - Skips frames based on config["frame_skip"].
    - Each step is applied in order; no frame is stored after yielding.
    """
    cfg = config or CONFIG

    # Initialize background subtractor once (maintains state across frames)
    bg_subtractor = None
    if cfg.get("use_background_subtraction", False):
        bg_subtractor = create_bg_subtractor(
            history=cfg.get("bg_history", 500),
            var_threshold=cfg.get("bg_var_threshold", 16),
        )

    # ── Iterate frames from folder ──────────────────────────────
    for frame_index, frame in load_frames(folder_path):

        # Step 2: Skip frames
        if not should_process_frame(frame_index, cfg.get("frame_skip", 3)):
            continue

        # Step 3: Resize
        frame = resize_frame(
            frame,
            width=cfg.get("resize_width", 640),
            height=cfg.get("resize_height", 480),
        )

        # Step 4: Apply ROI mask
        rois = cfg.get("rois", [])
        if rois:
            frame = apply_roi(frame, rois)

        # Step 5: Brightness & contrast
        frame = adjust_brightness_contrast(
            frame,
            alpha=cfg.get("alpha", 1.2),
            beta=cfg.get("beta", 15),
        )

        # Step 6: Noise reduction
        frame = reduce_noise(frame, kernel_size=cfg.get("blur_kernel", (3, 3)))

        # Step 7: Color conversion BGR → RGB
        frame = convert_bgr_to_rgb(frame)

        # Step 8: Optional background subtraction
        if bg_subtractor is not None:
            frame = apply_background_subtraction(frame, bg_subtractor)

        # Yield one processed frame — caller decides what to do with it
        yield frame_index, frame
