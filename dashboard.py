"""
dashboard.py
------------
Streamlit dashboard for the Traffic AI pipeline.
"""

import os
import sys
import time
import threading
import queue
import json
import traceback
from datetime import datetime
from pathlib import Path
from collections import deque, defaultdict
from typing import Optional

from dotenv import load_dotenv
load_dotenv(override=True)

import cv2
import numpy as np
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx
import plotly.graph_objects as go

# ─── Page config — MUST be first Streamlit call ─────────────────────────────
st.set_page_config(
    page_title="Traffic AI · Command Center",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Inject custom CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Barlow+Condensed:wght@300;600;800&display=swap');

  html, body, [class*="css"] {
    font-family: 'Barlow Condensed', sans-serif;
    background: #f8fafc;
    color: #0f172a;
  }

  /* Header */
  .dash-header {
    background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%);
    border-bottom: 2px solid #0ea5e9;
    padding: 18px 28px 14px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 16px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
  }
  .dash-title {
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: 2px;
    color: #0ea5e9;
    text-transform: uppercase;
    margin: 0;
    line-height: 1;
  }

  /* Metric cards */
  .metric-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: all 0.2s;
    box-shadow: 0 2px 4px rgba(0,0,0,0.02);
  }
  .metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--accent, #0ea5e9);
  }
  .metric-card:hover { 
    border-color: #0ea5e9;
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.05);
  }
  .metric-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    color: var(--accent, #0ea5e9);
    line-height: 1;
  }
  .metric-label {
    font-size: 0.72rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #64748b;
    margin-top: 4px;
  }

  /* Compact variant */
  .metric-card.compact {
    padding: 8px 12px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    height: 60px;
  }
  .metric-card.compact .metric-val {
    font-size: 1.4rem;
  }
  .metric-card.compact .metric-label {
    font-size: 0.6rem;
    letter-spacing: 1px;
    margin-top: 2px;
  }

  /* Signal lights */
  .signal-row {
    display: flex;
    gap: 8px;
    justify-content: center;
    flex-wrap: nowrap;
    padding: 4px 0;
  }
  .signal-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 100px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    border: 1.5px solid transparent;
    white-space: nowrap;
  }
  .sig-green  { background: #dcfce7; border-color: #22c55e; color: #166534; }
  .sig-yellow { background: #fef9c3; border-color: #eab308; color: #854d0e; }
  .sig-red    { background: #fee2e2; border-color: #ef4444; color: #991b1b; }
  .sig-dot { width: 8px; height: 8px; border-radius: 50%; }
  .sig-green .sig-dot  { background: #22c55e; box-shadow: 0 0 4px #22c55e; }
  .sig-yellow .sig-dot { background: #eab308; box-shadow: 0 0 4px #eab308; }
  .sig-red .sig-dot    { background: #ef4444; box-shadow: 0 0 4px #ef4444; }

  /* Alert banner */
  .alert-banner {
    background: #fee2e2;
    border: 1.5px solid #ef4444;
    border-radius: 8px;
    padding: 10px 16px;
    margin: 6px 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #b91c1c;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .emergency-banner {
    background: #ffedd5;
    border: 1.5px solid #f97316;
    border-radius: 8px;
    padding: 10px 16px;
    margin: 6px 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #9a3412;
  }

  /* Section headers */
  .section-head {
    font-size: 0.65rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #64748b;
    margin: 16px 0 8px;
    padding-bottom: 4px;
    border-bottom: 1px solid #e2e8f0;
  }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: #f1f5f9;
    border-right: 1px solid #e2e8f0;
  }
  section[data-testid="stSidebar"] * { color: #475569 !important; }
  section[data-testid="stSidebar"] h1,
  section[data-testid="stSidebar"] h2,
  section[data-testid="stSidebar"] h3 { color: #1e293b !important; }

  /* Log table */
  .log-entry {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    padding: 4px 8px;
    border-bottom: 1px solid #f1f5f9;
    color: #475569;
  }
  .log-entry.crash  { color: #dc2626; font-weight: 700; }
  .log-entry.emerg  { color: #ea580c; font-weight: 700; }
  .log-entry.normal { color: #475569; }

  /* Lane stats */
  .lane-stat-card {
    text-align: center;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 10px 4px;
    transition: all 0.2s;
  }
  .lane-stat-card:hover {
    border-color: #cbd5e1;
    background: #f8fafc;
  }

  /* Sidebar Buttons */
  div[data-testid="stSidebar"] button {
    height: 3.5rem !important;
    font-weight: 800 !important;
    font-size: 1.1rem !important;
    letter-spacing: 1px !important;
  }

  /* Hide default streamlit chrome except header (needed for sidebar toggle) */
  #MainMenu, footer { visibility: hidden; }
  header[data-testid="stHeader"] { background: transparent !important; }
  .block-container { padding-top: 0 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  SESSION STATE DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────
def _init_state(force_reset=False):
    defaults = {
        "running":          False,
        "stop_flag":        False,
        "frame_rgb":        None,
        "lane_counts":      {},
        "signal_output":    None,
        "stats": {
            "total_frames":     0,
            "total_vehicles":   0,
            "total_collisions": 0,
            "total_emergency":  0,
            "direction_counts": {},
            "lane_totals":      {},
            "seen_vehicle_ids": set(),
            "seen_entries":     set(), # Essential for new counting logic
            "seen_emergency_vehicle_ids": set(),
        },
        "event_log":        deque(maxlen=60),
        "fps_deque":        deque(maxlen=20),
        "current_fps":      0.0,
        "frames_processed": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state or force_reset:
            st.session_state[k] = v
    if force_reset:
        st.session_state.pop("id_frame_counts", None)
        st.session_state.pop("prev_emergency_lanes", None)

_init_state()

# ─────────────────────────────────────────────────────────────────────────────
#  PIPELINE THREAD — runs the actual AI pipeline in background
# ─────────────────────────────────────────────────────────────────────────────
def pipeline_thread(source_path: str, config: dict):
    print("=" * 55)
    print("  TRAFFIC ANALYSIS PIPELINE")
    print("=" * 55)
    print(f"  Source : {source_path}\n")

    # Local accumulators for exact summary matching
    total_frames = 0
    total_vehicles = 0
    emergency_frames = []
    total_emergency_detections = 0
    max_emergency_per_frame = 0
    emergency_vehicle_ids = set()
    emergency_lane_counts = {}
    lane_totals = {}
    direction_counts = {}
    seen_vehicle_ids = set()
    seen_in_lane = {}
    seen_in_direction = {}

    collision_logger = None
    signal_logger = None
    data_logger = None

    try:
        from pipeline import run_pipeline, build_frame_output
        from preprocessing import CONFIG as DEFAULT_PREPROCESS_CONFIG
        from detector import DETECTOR_CONFIG, VehicleDetector
        from tracker import TRACKER_CONFIG, CentroidTracker
        from lane_mapper import LANE_CONFIG, LaneMapper
        from emergency_detector import EmergencyLightDetector
        from collision_detector import CollisionDetector, CollisionLogger
        from signal_logger import SignalLogger
        from data_logger import DataLogger
        from crash_detector import CrashDetector
        from alert_dispatcher import AlertDispatcher
        from signal_controller import SignalController, SIGNAL_CONFIG

        print("[Pipeline] Initialising modules...")

        preprocess_cfg = {
            **DEFAULT_PREPROCESS_CONFIG,
            "resize_width":  1280,
            "resize_height": 720,
            "frame_skip":    config.get("frame_skip", 3),
            "rois":          [],
            "alpha":         config.get("alpha", 1.2),
            "beta":          config.get("beta", 15),
            "blur_kernel":   (3, 3),
            "use_clahe":     config.get("use_clahe", False),
            "clahe_clip_limit": 2.0,
            "clahe_tile_grid":  (8, 8)
        }
        detector_cfg = {
            **DETECTOR_CONFIG,
            "confidence_threshold": config.get("conf_thresh", 0.10),
            "imgsz": 640, # 640 is the sweet spot for YOLOv8 foreground detection
        }
        tracker_cfg = {
            **TRACKER_CONFIG,
            "max_distance": 100,
            "max_lost_frames": 15,
            "min_hits": 1,
        }
        
        detector    = VehicleDetector(detector_cfg)
        tracker     = CentroidTracker(tracker_cfg)
        lane_mapper = LaneMapper(LANE_CONFIG)
        em_detector = EmergencyLightDetector()
        col_detector= CollisionDetector()
        crash_det   = CrashDetector()
        alert_disp  = AlertDispatcher()
        sig_ctrl    = SignalController(SIGNAL_CONFIG)
        sig_ctrl.set_frame_rate(fps=30, frame_skip=preprocess_cfg.get("frame_skip", 3))

        collision_logger = CollisionLogger(log_file="collision_log.txt")
        signal_logger     = SignalLogger(log_file="signal_log.txt")
        data_logger       = DataLogger()

        print("[Pipeline] All modules ready. Starting frame processing...\n")

        from preprocessing import apply_clahe, convert_bgr_to_rgb, reduce_noise, adjust_brightness_contrast

        def frames_from_folder(folder_path):
            valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")
            filenames = sorted(f for f in os.listdir(folder_path) if f.lower().endswith(valid_ext))
            for i, f in enumerate(filenames):
                if st.session_state.stop_flag: break
                if i % preprocess_cfg["frame_skip"] != 0: continue
                filepath = os.path.join(folder_path, f)
                frame = cv2.imread(filepath)
                if frame is not None:
                    yield i, cv2.resize(frame, (1280, 720))

        t_prev = time.time()
        for idx, frame_bgr in frames_from_folder(source_path):
            # --- Professional Vision Pipeline ---
            from preprocessing import preprocess_for_yolo
            frame_rgb = preprocess_for_yolo(frame_bgr, force_clahe=preprocess_cfg.get("use_clahe"))
            # ------------------------------------

            active_tracks = detector.detect(frame_rgb, idx)
            
            frame_out = build_frame_output(
                idx, frame_bgr, frame_rgb,
                active_tracks, lane_mapper,
                em_detector, col_detector, crash_det
            )
            
            crash_report = crash_det.update(frame_out)
            is_new_crash = False
            if crash_report:
                is_new_crash = alert_disp.dispatch(crash_report, frame_out.get("debug_frame"))
                
            gated_collisions = [{"lane": crash_report["lane"]}] if crash_report else []
            signal_out = sig_ctrl.update(frame_out["lane_counts"], frame_out["emergency_lane"], gated_collisions, idx)

            # Log frame to logging systems
            if signal_out:
                signal_logger.log(idx, signal_out)
            collision_logger.log(frame_out.get("collisions", []))
            data_logger.log(frame_out, signal_out)

            # --- Print frame-level logging ---
            # 1. Crash print matching example_usage.py
            if crash_report:
                sev   = crash_report["severity"].upper()
                score = crash_report["score"]
                lane  = crash_report["lane"]
                print(f"  Frame {idx:>5} | [CRASH {sev}] score={score} "
                      f"lane={lane} signals={crash_report['signals']}")

            emerg_lane = frame_out.get("emergency_lane", [])
            emerg_ids  = frame_out.get("emergency_veh_ids", set())

            # --- Update local accumulators for the summary ---
            total_frames += 1
            for v in frame_out["vehicles"]:
                vid = v.get("id")
                if vid is None:
                    continue

                # Unique global count
                if vid not in seen_vehicle_ids:
                    seen_vehicle_ids.add(vid)
                    total_vehicles += 1

                # Unique direction count
                d = v.get("direction") or "unknown"
                if d not in seen_in_direction:
                    seen_in_direction[d] = set()
                if vid not in seen_in_direction[d]:
                    seen_in_direction[d].add(vid)
                    direction_counts[d] = direction_counts.get(d, 0) + 1

                # Unique lane count
                lane = v.get("lane") or "unknown"
                if lane not in seen_in_lane:
                    seen_in_lane[lane] = set()
                if vid not in seen_in_lane[lane]:
                    seen_in_lane[lane].add(vid)
                    lane_totals[lane] = lane_totals.get(lane, 0) + 1

            if emerg_lane:
                if idx not in emergency_frames:
                    emergency_frames.append(idx)
                total_emergency_detections += len(emerg_lane)
                for lane in emerg_lane:
                    emergency_lane_counts[lane] = emergency_lane_counts.get(lane, 0) + 1
                max_emergency_per_frame = max(max_emergency_per_frame, len(emerg_lane))
                for vid in emerg_ids:
                    emergency_vehicle_ids.add(vid)

            # Update State
            st.session_state.frame_rgb = frame_out.get("debug_frame").copy()
            st.session_state.lane_counts = frame_out["lane_counts"]
            st.session_state.signal_output = signal_out.to_dict() if signal_out else None
            st.session_state.frames_processed = idx
            
            t_now = time.time()
            fps = 1.0 / max(t_now - t_prev, 1e-6)
            t_prev = t_now
            st.session_state.fps_deque.append(fps)
            st.session_state.current_fps = sum(st.session_state.fps_deque) / len(st.session_state.fps_deque)

            # --- High-Accuracy Validation Engine ---
            s = st.session_state.stats
            s["total_frames"] = total_frames
            
            # 1. Unique emergency vehicle tracking
            if "seen_emergency_vehicle_ids" not in s:
                s["seen_emergency_vehicle_ids"] = set()
            for vid in frame_out.get("emergency_veh_ids", set()):
                if vid is not None and vid != -1:
                    s["seen_emergency_vehicle_ids"].add(vid)
            
            # Update emergency count precisely based on unique tracked emergency vehicles,
            # fallback to 1 if emergency lights are active but vehicle matching is still pending.
            # We keep this cumulative so it doesn't drop back to 0 at the end of the run.
            if len(s["seen_emergency_vehicle_ids"]) > 0:
                s["total_emergency"] = len(s["seen_emergency_vehicle_ids"])
            elif frame_out.get("emergency_lane"):
                s["total_emergency"] = max(s.get("total_emergency", 0), 1)
            
            # 2. Unique vehicle counting & lane/direction tracking
            for v in frame_out["vehicles"]:
                vid = v.get("id")
                lane = v.get("lane")
                if vid is not None and vid != -1 and lane:
                    # Unique vehicle IDs
                    if vid not in s["seen_vehicle_ids"]:
                        s["seen_vehicle_ids"].add(vid)
                        s["total_vehicles"] += 1
                    
                    # Cumulative lane totals
                    lane_key = f"{vid}_{lane}"
                    if lane_key not in s.get("seen_entries", set()):
                        if "seen_entries" not in s: s["seen_entries"] = set()
                        s["seen_entries"].add(lane_key)
                        s["lane_totals"][lane] = s["lane_totals"].get(lane, 0) + 1
                    
                    # Unique directions
                    d = v.get("direction") or "unknown"
                    if "seen_in_direction" not in s:
                        s["seen_in_direction"] = {}
                    if d not in s["seen_in_direction"]:
                        s["seen_in_direction"][d] = set()
                    if vid not in s["seen_in_direction"][d]:
                        s["seen_in_direction"][d].add(vid)
                        s["direction_counts"][d] = s["direction_counts"].get(d, 0) + 1
            
            if is_new_crash: s["total_collisions"] += 1
            st.session_state.stats = s
            # ----------------------------------------

            # Log events (deduplicated to prevent spam)
            ts = datetime.now().strftime("%H:%M:%S")
            if is_new_crash:
                st.session_state.event_log.appendleft({"type": "crash", "ts": ts, "msg": f"CRASH: {crash_report['lane']} ({crash_report['severity']})"})
            
            if "prev_emergency_lanes" not in st.session_state:
                st.session_state.prev_emergency_lanes = set()
            
            current_emerg = set(frame_out.get("emergency_lane", []))
            new_emerg = current_emerg - st.session_state.prev_emergency_lanes
            
            for em in new_emerg:
                st.session_state.event_log.appendleft({"type": "emerg", "ts": ts, "msg": f"EMERGENCY: {em}"})
            
            st.session_state.prev_emergency_lanes = current_emerg

    except Exception as e:
        traceback.print_exc()
    finally:
        st.session_state.running = False
        
        # --- Print end-of-run summary matching example_usage.py ---
        print("\n" + "═" * 55)
        print("  PIPELINE SUMMARY (Phase 1: Traffic Analysis)")
        print("═" * 55)
        print(f"  Frames processed        : {total_frames}")
        print(f"  Total vehicle obs       : {total_vehicles}")
        print()

        print(f"  Emergency events")
        print(f"  ├─ Frames with emergency : {len(emergency_frames)}")
        if emergency_frames:
            print(f"  ├─ Frame IDs             : {emergency_frames}")
        print(f"  ├─ Total detections      : {total_emergency_detections}")
        print(f"  ├─ Peak lanes at once    : {max_emergency_per_frame}")
        id_str = f"  {sorted(emergency_vehicle_ids)}" if emergency_vehicle_ids else "  none"
        print(f"  └─ Emergency vehicle IDs : {len(emergency_vehicle_ids)}{id_str}")
        if emergency_lane_counts:
            print()
            print("  Emergency lane breakdown:")
            for lane, count in sorted(emergency_lane_counts.items(), key=lambda x: -x[1]):
                bar = "█" * min(count, 30)
                print(f"    {lane:<12} {count:>4} frames  {bar}")
        print()

        print("  Cumulative lane counts:")
        for lane, count in sorted(lane_totals.items()):
            bar = "█" * min(count, 40)
            print(f"    {lane:<10} {count:>5}  {bar}")
        print()

        print("  Vehicle directions:")
        for direction, count in sorted(direction_counts.items(), key=lambda x: -x[1]):
            print(f"    {direction:<12} {count}")
        print("═" * 55)

        if collision_logger is not None:
            collision_logger.print_summary()

        if signal_logger is not None:
            signal_logger.print_summary()
            signal_logger.export_json("signal_stats.json")

        if data_logger is not None:
            data_logger.close()

        print("\n[Complete] All summaries printed.")
        print("  • collision_log.txt   — per-collision events")
        print("  • signal_log.txt      — per-frame signal state")
        print("  • signal_stats.json   — signal summary statistics")
        print("  • traffic_log.csv     — Phase 3 DQN training data")
        print("  • alerts_log.csv      — confirmed crash alerts")
        print("  • alerts/             — crash snapshot images")

# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Source")
    
    # Initialize path history
    if "path_history" not in st.session_state:
        st.session_state.path_history = []
        env_path = os.getenv("FRAMES_FOLDER_PATH", "")
        if env_path: st.session_state.path_history.append(env_path)

    # Dropdown for history
    hist_options = ["None"] + st.session_state.path_history
    quick_select = st.selectbox("Quick Select", options=hist_options, index=0)
    
    # Default value for text input
    default_val = quick_select if quick_select != "None" else (st.session_state.path_history[0] if st.session_state.path_history else "")
    source_path = st.text_input("Frames folder path", value=default_val)
    
    st.markdown("### Detection")
    conf_thresh = 0.10 # Lowered to catch low-confidence silver cars
    frame_skip = st.number_input("Frame skip", 1, 30, 3)
    
    vision_mode = st.selectbox(
        "Vision Mode",
        ["Adaptive (Auto)", "Daytime (Clean RGB)", "Night Mode (CLAHE)"],
        index=0,
        help="Adaptive mode auto-detects lighting and applies CLAHE only when dark."
    )
    
    # Map selection to force_clahe parameter
    force_clahe = None
    if vision_mode == "Daytime (Clean RGB)":
        force_clahe = False
    elif vision_mode == "Night Mode (CLAHE)":
        force_clahe = True

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    if col1.button("▶ START", use_container_width=True, type="primary"):
        # Update path history on start
        if source_path:
            if source_path in st.session_state.path_history:
                st.session_state.path_history.remove(source_path)
            st.session_state.path_history.insert(0, source_path)
            st.session_state.path_history = st.session_state.path_history[:3] # Keep last 3 paths
            
        _init_state(force_reset=True)
        from pipeline import build_frame_output
        build_frame_output._track_history = {}
        st.session_state.running = True
        st.session_state.stop_flag = False
        t = threading.Thread(target=pipeline_thread, args=(source_path, {"conf_thresh": conf_thresh, "frame_skip": frame_skip, "use_clahe": force_clahe}), daemon=True)
        add_script_run_ctx(t)
        t.start()
        st.rerun()
    
    if col2.button("⏹ STOP", use_container_width=True):
        st.session_state.stop_flag = True
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
#  UI RENDERING
# ─────────────────────────────────────────────────────────────────────────────
@st.fragment(run_every=0.12)
def render_ui():
    status = "🟢 LIVE" if st.session_state.running else "🔴 STOPPED"
    st.markdown(f"""
    <div class="dash-header">
      <div><p class="dash-title">🚦 Traffic Command Center</p></div>
      <div style="margin-left:auto; text-align:right; font-family:'JetBrains Mono'; color:#64748b;">
        {status} | FPS: {st.session_state.current_fps:.1f} | Frames: {st.session_state.frames_processed}
      </div>
    </div>
    """, unsafe_allow_html=True)

    t1, t2 = st.columns(2)
    with t1:
        st.markdown('<p class="section-head">Signal State & Timings</p>', unsafe_allow_html=True)
        sig = st.session_state.signal_output
        if sig:
            # Timing and Phase info matching example_usage.py exactly
            phase_id = sig.get("phase_id", 0)
            phase_name = sig.get("phase_name", "Unknown")
            time_left = int(sig.get("time_until_next", 0))
            is_yellow = sig.get("is_yellow_mode", False)
            reason = sig.get("override_reason", "standard_adaptive")
            
            mode_color = "#166534"
            mode_bg = "#dcfce7"
            if "collision" in reason:
                mode_color = "#991b1b"
                mode_bg = "#fee2e2"
            elif "emergency" in reason:
                mode_color = "#9a3412"
                mode_bg = "#ffedd5"
                
            time_label = f"YELLOW {time_left}s" if is_yellow else f"GREEN {time_left}s"
            time_class = "sig-yellow" if is_yellow else "sig-green"
            
            st.markdown(f"""
            <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:10px; padding:12px; font-family:'JetBrains Mono', monospace; font-size:0.75rem; color:#334155; margin-bottom:12px;">
                <div style="font-weight:700; color:#1e293b; font-size:0.8rem; margin-bottom:6px;">🔹 Phase {phase_id}: {phase_name}</div>
                <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
                    <span class="signal-pill {time_class}" style="padding:2px 8px; font-size:0.65rem;"><span class="sig-dot"></span>{time_label}</span>
                    <span style="font-size:0.62rem; color:{mode_color}; background:{mode_bg}; border-radius:100px; padding:2px 8px; font-weight:700;">{reason.upper()}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Render individual lane signal pills
            pills = ""
            for lane in ["top_road", "bottom_road", "left_road", "right_road"]:
                color = "sig-green" if lane in sig.get("active_lanes", []) else ("sig-yellow" if lane in sig.get("yellow_lanes", []) else "sig-red")
                full_name = lane.replace("_road", "").upper()
                pills += f'<span class="signal-pill {color}"><span class="sig-dot"></span>{full_name}</span>'
            st.markdown(f'<div class="signal-row" style="justify-content:flex-start; gap:6px;">{pills}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="color:#94a3b8; font-size:0.75rem; margin-top:8px;">SIGNAL OFFLINE</div>', unsafe_allow_html=True)

    with t2:
        st.markdown('<p class="section-head">System Stats</p>', unsafe_allow_html=True)
        s = st.session_state.stats
        m = st.columns(4)
        m[0].markdown(f'<div class="metric-card compact"><div class="metric-val">{s["total_frames"]}</div><div class="metric-label">FRAMES</div></div>', unsafe_allow_html=True)
        m[1].markdown(f'<div class="metric-card compact"><div class="metric-val">{s["total_vehicles"]}</div><div class="metric-label">VEHICLES</div></div>', unsafe_allow_html=True)
        m[2].markdown(f'<div class="metric-card compact"><div class="metric-val">{s["total_collisions"]}</div><div class="metric-label">CRASHES</div></div>', unsafe_allow_html=True)
        m[3].markdown(f'<div class="metric-card compact"><div class="metric-val">{s["total_emergency"]}</div><div class="metric-label">EMERGENCY</div></div>', unsafe_allow_html=True)

    st.markdown('<hr style="margin:15px 0; border:0; border-top:1px solid #e2e8f0;">', unsafe_allow_html=True)

    c1, c2 = st.columns([2.2, 1])
    with c1:
        st.markdown('<p class="section-head">Live Intelligence Feed</p>', unsafe_allow_html=True)
        if st.session_state.frame_rgb is not None:
            st.image(st.session_state.frame_rgb, width='stretch', channels="RGB")
        else:
            st.markdown('<div style="width:100%; aspect-ratio:16/9; background:#f1f5f9; border-radius:12px; display:flex; align-items:center; justify-content:center; color:#94a3b8; font-family:\'JetBrains Mono\';">SYSTEM OFFLINE</div>', unsafe_allow_html=True)
        
        st.markdown('<p class="section-head">Real-time Event Log</p>', unsafe_allow_html=True)
        log_html = '<div style="max-height:200px; overflow-y:auto; background:#ffffff; border:1px solid #e2e8f0; border-radius:8px; padding:8px;">'
        if not st.session_state.event_log:
            log_html += '<div style="color:#94a3b8; font-size:0.75rem; text-align:center;">No events recorded.</div>'
        else:
            for ev in list(st.session_state.event_log):
                log_html += f'<div class="log-entry {ev["type"]}">[{ev["ts"]}] {ev["msg"]}</div>'
        log_html += '</div>'
        st.markdown(log_html, unsafe_allow_html=True)

    with c2:
        st.markdown('<p class="section-head">Live Lane Density</p>', unsafe_allow_html=True)
        lc = st.session_state.lane_counts
        colors = ["#eab308", "#22c55e", "#2563eb", "#ef4444"]
        cols = st.columns(2)
        for i, l in enumerate(["top_road", "bottom_road", "left_road", "right_road"]):
            cols[i%2].markdown(f'<div class="lane-stat-card"><div style="color:{colors[i]}; font-size:1.6rem; font-weight:700;">{lc.get(l,0)}</div><div style="font-size:0.6rem; color:#64748b;">{l.replace("_road","").upper()}</div></div>', unsafe_allow_html=True)

        st.markdown('<p class="section-head">Historical Lane Totals</p>', unsafe_allow_html=True)
        lt = st.session_state.stats["lane_totals"]
        cols_h = st.columns(2)
        for i, l in enumerate(["top_road", "bottom_road", "left_road", "right_road"]):
            cols_h[i%2].markdown(f'<div class="lane-stat-card"><div style="color:{colors[i]}; font-size:1.4rem; font-weight:700;">{lt.get(l,0)}</div><div style="font-size:0.55rem; color:#64748b;">{l.replace("_road","").upper()} TOTAL</div></div>', unsafe_allow_html=True)

        st.markdown('<p class="section-head">Dispatch Alerts</p>', unsafe_allow_html=True)
        if st.session_state.frames_processed > 0:
            s = st.session_state.stats
            if s.get("total_collisions", 0) > 0:
                st.markdown('<div class="alert-banner" style="justify-content:center; text-align:center;">🚨 ACCIDENT ALERT: CRASH DETECTED</div>', unsafe_allow_html=True)
            elif s.get("total_emergency", 0) > 0:
                st.markdown('<div class="emergency-banner" style="justify-content:center; text-align:center;">⚠️ EMERGENCY VEHICLE ACTIVE</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div style="color:#166534; background:#dcfce7; border-radius:8px; padding:10px; text-align:center; font-size:0.75rem; font-weight:700;">✓ SYSTEM CLEAR</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="color:#94a3b8; font-size:0.75rem; text-align:center;">Waiting for stream...</div>', unsafe_allow_html=True)

render_ui()
