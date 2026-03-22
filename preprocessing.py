"""
preprocessing.py
----------------
Memory-efficient preprocessing pipeline for traffic video frames.
Designed for YOLO compatibility, CPU-only, low RAM usage (<2GB).
Frames are processed one-by-one using Python generators.

Updates for night-time / fisheye camera support:
  - Added CLAHE contrast enhancement (Step 5a) for night footage
  - Added fisheye undistortion (Step 0) — disabled by default
  - Lowered default ROI top boundary from y=100 to y=60
  - Added use_clahe toggle in CONFIG
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
    # FIX: lowered top boundary from y=100 to y=60 so upper-frame vehicles
    # (fire trucks, police cars, ambulances) are no longer cut off.
    "rois": [
        (0, 60, 640, 480),
    ],

    # Brightness / Contrast (cv2.convertScaleAbs)
    # alpha: contrast multiplier (1.0 = no change, >1 = more contrast)
    # beta:  brightness offset   (0 = no change, positive = brighter)
    "alpha": 1.2,
    "beta": 15,

    # CLAHE (Contrast Limited Adaptive Histogram Equalisation)
    # FIX: replaces flat alpha/beta for night footage.
    # Equalises contrast locally per tile — handles blown-out lights AND
    # dark road areas in the same frame, which convertScaleAbs cannot do.
    # Set use_clahe=True for night / low-light footage.
    # Set use_clahe=False for daytime (uses alpha/beta instead).
    "use_clahe": True,
    "clahe_clip_limit": 2.0,       # higher = more contrast, more noise risk
    "clahe_tile_grid": (8, 8),     # tile size — (8,8) is standard

    # Gaussian Blur kernel size — must be odd (e.g. 3, 5, 7)
    "blur_kernel": (3, 3),

    # Fisheye / wide-angle lens undistortion
    # FIX: wide-angle cameras warp vehicles near edges — YOLO confidence
    # drops significantly on distorted shapes.
    # Set use_undistort=True and fill in your camera's calibration values.
    "use_undistort": False,
    # Camera matrix [fx, 0, cx], [0, fy, cy], [0, 0, 1]
    # Replace with values from cv2.calibrateCamera() on your camera.
    "camera_matrix": [
        [600.0,   0.0, 320.0],
        [  0.0, 600.0, 240.0],
        [  0.0,   0.0,   1.0],
    ],
    # Distortion coefficients [k1, k2, p1, p2, k3]
    # Negative k1/k2 = barrel distortion (fisheye). Replace with your values.
    "dist_coeffs": [-0.3, 0.1, 0.0, 0.0, 0.0],

    # Background subtraction toggle
    "use_background_subtraction": False,

    # Background subtractor parameters
    "bg_history": 500,
    "bg_var_threshold": 16,
}


# ─────────────────────────────────────────────
#  STEP 0 — OPTIONAL: FISHEYE UNDISTORTION
# ─────────────────────────────────────────────
def undistort_frame(
    frame: np.ndarray,
    camera_matrix: list,
    dist_coeffs: list,
) -> np.ndarray:
    """
    Corrects fisheye / wide-angle lens distortion.

    Vehicles near frame edges are warped by fisheye lenses — their
    bounding box shapes no longer match what YOLO was trained on,
    causing confidence to drop and detections to be missed.

    camera_matrix: 3x3 intrinsic matrix from cv2.calibrateCamera()
    dist_coeffs:   [k1, k2, p1, p2, k3] distortion coefficients

    To calibrate your camera: photograph a checkerboard from multiple
    angles and run cv2.calibrateCamera(). Use those values here.
    """
    K = np.array(camera_matrix, dtype=np.float32)
    D = np.array(dist_coeffs,   dtype=np.float32)
    h, w = frame.shape[:2]
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=1)
    return cv2.undistort(frame, K, D, None, new_K)


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
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")

    filenames = sorted(
        f for f in os.listdir(folder_path)
        if f.lower().endswith(valid_ext)
    )

    if not filenames:
        raise FileNotFoundError(f"No image frames found in: {folder_path}")

    for idx, filename in enumerate(filenames):
        filepath = os.path.join(folder_path, filename)
        frame = cv2.imread(filepath)

        if frame is None:
            print(f"[WARNING] Could not read frame: {filepath} — skipping.")
            continue

        yield idx, frame


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
        return frame

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for (x1, y1, x2, y2) in rois:
        mask[y1:y2, x1:x2] = 255
    return cv2.bitwise_and(frame, frame, mask=mask)


# ─────────────────────────────────────────────
#  STEP 5 — BRIGHTNESS & CONTRAST (daytime)
# ─────────────────────────────────────────────
def adjust_brightness_contrast(
    frame: np.ndarray, alpha: float = 1.2, beta: int = 15
) -> np.ndarray:
    """
    Adjusts brightness and contrast using a global linear transform.
    Best for daytime footage with consistent lighting.

    alpha: contrast multiplier (1.0 = unchanged, >1 = more contrast)
    beta:  brightness offset   (0 = unchanged, positive = brighter)

    Uses cv2.convertScaleAbs which clips values to [0, 255].
    For night footage, use apply_clahe() instead — it handles local
    contrast variation that this global approach cannot.
    """
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)


# ─────────────────────────────────────────────
#  STEP 5a — CLAHE (night / low-light footage)
# ─────────────────────────────────────────────
def apply_clahe(
    frame: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid: tuple = (8, 8),
) -> np.ndarray:
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalisation).

    Unlike global alpha/beta which applies the same transform to every pixel,
    CLAHE divides the image into tiles and equalises each independently.
    This means it can simultaneously:
      - Brighten dark road areas where vehicles are hard to see
      - Suppress blown-out emergency lights (blue/red flashing)
      - Recover vehicle edge detail under mixed artificial lighting

    Works on the L (lightness) channel in LAB colour space so colour
    information is preserved — YOLO's colour features remain valid.

    clip_limit:  max contrast amplification per tile (2.0 = standard)
    tile_grid:   (cols, rows) of tiles — (8,8) works for 640x480
    """
    # Convert BGR → LAB (L = lightness, A/B = colour)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Apply CLAHE only to lightness — colour channels untouched
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_enhanced = clahe.apply(l_channel)

    # Merge back and convert to BGR
    enhanced_lab = cv2.merge([l_enhanced, a_channel, b_channel])
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


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
        detectShadows=False,
    )


def apply_background_subtraction(
    frame: np.ndarray, bg_subtractor
) -> np.ndarray:
    """
    Applies MOG2 background subtraction to isolate moving objects.

    bg_subtractor must be the same instance across frames to learn background.
    """
    fg_mask = bg_subtractor.apply(frame)
    fg_mask_3ch = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
    return cv2.bitwise_and(frame, fg_mask_3ch)


# ─────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────
def preprocess_pipeline(folder_path: str, config: dict = None):
    """
    Main generator pipeline. Applies all preprocessing steps sequentially.

    Yields: (frame_index: int, preprocessed_frame: np.ndarray)

    Step order:
      0. Undistort (optional) — fisheye/wide-angle correction
      1. Load frame from disk
      2. Frame skip check
      3. Resize to target dimensions
      4. ROI mask — zero out non-road regions
      5. Contrast enhancement — CLAHE (night) or alpha/beta (day)
      6. Gaussian blur — remove compression noise
      7. BGR → RGB — required for YOLO
      8. Background subtraction (optional)
    """
    cfg = config or CONFIG

    # Prepare undistortion matrices once (expensive to recompute per frame)
    undistort_enabled = cfg.get("use_undistort", False)
    K = D = None
    if undistort_enabled:
        K = np.array(cfg.get("camera_matrix"), dtype=np.float32)
        D = np.array(cfg.get("dist_coeffs"),   dtype=np.float32)

    # Prepare CLAHE instance once (reused across frames)
    use_clahe = cfg.get("use_clahe", False)
    clahe_obj = None
    if use_clahe:
        clahe_obj = cv2.createCLAHE(
            clipLimit=cfg.get("clahe_clip_limit", 2.0),
            tileGridSize=cfg.get("clahe_tile_grid", (8, 8)),
        )

    # Initialize background subtractor once
    bg_subtractor = None
    if cfg.get("use_background_subtraction", False):
        bg_subtractor = create_bg_subtractor(
            history=cfg.get("bg_history", 500),
            var_threshold=cfg.get("bg_var_threshold", 16),
        )

    for frame_index, frame in load_frames(folder_path):

        # Step 2: Skip frames
        if not should_process_frame(frame_index, cfg.get("frame_skip", 3)):
            continue

        # Step 0: Undistort fisheye (before resize — works at native resolution)
        if undistort_enabled and K is not None:
            frame = undistort_frame(
                frame,
                cfg.get("camera_matrix"),
                cfg.get("dist_coeffs"),
            )

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

        # Step 5: Contrast enhancement
        if use_clahe:
            # Night / mixed lighting — CLAHE handles local contrast variation
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = clahe_obj.apply(l)
            frame = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        else:
            # Daytime — simple global brightness/contrast boost
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

        yield frame_index, frame