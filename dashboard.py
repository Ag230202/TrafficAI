"""
dashboard.py
------------
Streamlit dashboard for the Traffic AI pipeline.

Runs the full pipeline on a video or frames folder and displays:
  • Live annotated video feed
  • Per-lane vehicle counts
  • Traffic signal state (coloured indicators per lane)
  • Emergency vehicle alerts
  • Crash / collision events
  • Pipeline statistics (totals, directions, emergency summary)

Launch:
    streamlit run dashboard.py

Dependencies (install once):
    pip install streamlit ultralytics opencv-python-headless torch torchvision plotly
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
load_dotenv()

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

  /* Hide default streamlit chrome */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 0 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  SESSION STATE DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "running":          False,
        "stop_flag":        False,
        "frame_rgb":        None,
        "lane_counts":      {},
        "signal_output":    None,
        "crash_report":     None,
        "emergency_lanes":  [],
        "collisions":       [],
        "stats": {
            "total_frames":     0,
            "total_vehicles":   0,
            "total_collisions": 0,
            "total_emergency":  0,
            "direction_counts": {},
            "lane_totals":      {},
            "seen_vehicle_ids": set(),
            "seen_in_lane":     {},
            "seen_in_direction":{},
        },
        "event_log":        deque(maxlen=60),
        "count_history":    defaultdict(lambda: deque(maxlen=80)),
        "fps_deque":        deque(maxlen=20),
        "current_fps":      0.0,
        "frames_processed": 0,
        "last_crash_time":  0.0,
        "last_emerg_time":  0.0,
        "persisted_crash":  None,
        "persisted_emerg":  [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ─────────────────────────────────────────────────────────────────────────────
#  PIPELINE THREAD — runs the actual AI pipeline in background
# ─────────────────────────────────────────────────────────────────────────────
def pipeline_thread(source_path: str, config: dict):
    """
    Background worker that runs the full Traffic AI pipeline.
    Writes results into st.session_state (thread-safe for simple assignments).
    """
    print(f"\\n[INFO] Starting AI Pipeline Thread...")
    print(f"[INFO] Source: {source_path}")
    try:
        # ── Lazy imports (avoid loading at module level) ────────────────
        from pipeline import run_pipeline, build_frame_output
        from preprocessing import CONFIG as DEFAULT_PREPROCESS_CONFIG
        from detector import DETECTOR_CONFIG, VehicleDetector
        from tracker import TRACKER_CONFIG, CentroidTracker
        from lane_mapper import LANE_CONFIG, LaneMapper
        from emergency_detector import EmergencyLightDetector
        from collision_detector import CollisionDetector
        from crash_detector import CrashDetector
        from alert_dispatcher import AlertDispatcher
        from signal_controller import SignalController, SIGNAL_CONFIG

        # ── Build configs from dashboard settings ───────────────────────
        preprocess_cfg = {
            **DEFAULT_PREPROCESS_CONFIG,
            "resize_width":  1280,
            "resize_height": 720,
            "frame_skip":    config.get("frame_skip", 3),
            "rois":          [],
            "alpha":         config.get("alpha", 1.2),
            "beta":          config.get("beta", 15),
            "blur_kernel":   (3, 3),
            "use_clahe":     config.get("use_clahe", True),
            "clahe_clip_limit": 2.0,
            "clahe_tile_grid":  (8, 8),
            "use_background_subtraction": False,
        }

        detector_cfg = {
            **DETECTOR_CONFIG,
            "confidence_threshold": config.get("conf_thresh", 0.20),
            "imgsz": config.get("imgsz", 1280),
        }

        tracker_cfg = {
            **TRACKER_CONFIG,
            "max_distance":    100,
            "max_lost_frames": 8,
            "min_hits":        2,
        }

        lane_cfg = config.get("lane_config", LANE_CONFIG)

        signal_cfg = {
            **SIGNAL_CONFIG,
            "use_dqn":                  config.get("use_dqn", False),
            "enable_adaptive":          True,
            "emergency_preemption":     True,
            "enable_collision_override": True,
        }

        # ── Instantiate modules ─────────────────────────────────────────
        detector    = VehicleDetector(detector_cfg)
        tracker     = CentroidTracker(tracker_cfg)
        lane_mapper = LaneMapper(lane_cfg)
        em_detector = EmergencyLightDetector()
        col_detector= CollisionDetector()
        crash_det   = CrashDetector()
        alert_disp  = AlertDispatcher()
        sig_ctrl    = SignalController(signal_cfg)

        # ── Determine frame source ──────────────────────────────────────
        from preprocessing import (
            apply_roi, adjust_brightness_contrast, apply_clahe,
            reduce_noise, convert_bgr_to_rgb,
        )

        resize_w    = preprocess_cfg["resize_width"]
        resize_h    = preprocess_cfg["resize_height"]
        frame_skip  = preprocess_cfg["frame_skip"]
        use_clahe   = preprocess_cfg["use_clahe"]

        def preprocess_bgr(frame_bgr):
            frame_rgb = frame_bgr.copy()
            rois = preprocess_cfg.get("rois", [])
            if rois:
                frame_rgb = apply_roi(frame_rgb, rois)
            if use_clahe:
                frame_rgb = apply_clahe(frame_rgb,
                    clip_limit=preprocess_cfg.get("clahe_clip_limit", 2.0),
                    tile_grid=preprocess_cfg.get("clahe_tile_grid", (8, 8)))
            else:
                frame_rgb = adjust_brightness_contrast(frame_rgb,
                    alpha=preprocess_cfg.get("alpha", 1.2),
                    beta=preprocess_cfg.get("beta", 15))
            frame_rgb = reduce_noise(frame_rgb, preprocess_cfg.get("blur_kernel", (3, 3)))
            frame_rgb = convert_bgr_to_rgb(frame_rgb)
            return frame_rgb

        def frames_from_folder(folder_path):
            valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")
            filenames = sorted(
                f for f in os.listdir(folder_path)
                if f.lower().endswith(valid_ext)
            )
            for frame_index, filename in enumerate(filenames):
                if st.session_state.stop_flag:
                    break
                if frame_index % frame_skip != 0:
                    continue
                filepath = os.path.join(folder_path, filename)
                frame_bgr = cv2.imread(filepath)
                if frame_bgr is None:
                    continue
                frame_bgr = cv2.resize(frame_bgr, (resize_w, resize_h),
                                       interpolation=cv2.INTER_LINEAR)
                yield frame_index, frame_bgr

        frame_source = frames_from_folder(source_path)

        # ── Main loop ───────────────────────────────────────────────────
        t_prev = time.time()
        for frame_index, frame_bgr in frame_source:
            if st.session_state.stop_flag:
                print("[INFO] Stop signal received. Terminating pipeline thread...")
                break

            if frame_index % (frame_skip * 10) == 0:
                 print(f"[PROGRESS] Processing frame {frame_index}...")

            frame_rgb = preprocess_bgr(frame_bgr)

            raw_dets     = detector.detect(frame_rgb, frame_index)
            active_tracks= tracker.update(raw_dets)

            frame_out = build_frame_output(
                frame_index, frame_bgr, frame_rgb,
                active_tracks, lane_mapper,
                em_detector, col_detector,
            )

            # Crash detection
            crash_report = crash_det.update(frame_out)
            is_new_crash = False
            if crash_report:
                is_new_crash = alert_disp.dispatch(crash_report, frame_out.get("debug_frame"))

            # Signal control
            gated_collisions = [{"lane": crash_report["lane"]}] if crash_report else []
            signal_out = sig_ctrl.update(
                lane_counts=frame_out["lane_counts"],
                emergency_lane=frame_out["emergency_lane"],
                collisions=gated_collisions,
                frame_id=frame_index,
            )

            # FPS calc
            t_now = time.time()
            fps   = (1.0 / max(t_now - t_prev, 1e-6)) * frame_skip
            t_prev = t_now
            st.session_state.fps_deque.append(fps)
            st.session_state.current_fps = sum(st.session_state.fps_deque) / len(st.session_state.fps_deque)

            # Write to session state (simple assignment is thread-safe in CPython)
            debug = frame_out.get("debug_frame")
            if debug is not None:
                st.session_state.frame_rgb = debug.copy()

            st.session_state.lane_counts     = frame_out["lane_counts"]
            st.session_state.signal_output   = signal_out.to_dict() if signal_out else None
            st.session_state.crash_report    = crash_report
            if crash_report:
                st.session_state.last_crash_time = time.time()
                st.session_state.persisted_crash = crash_report

            st.session_state.emergency_lanes = frame_out.get("emergency_lane", [])
            if st.session_state.emergency_lanes:
                st.session_state.last_emerg_time = time.time()
                st.session_state.persisted_emerg = st.session_state.emergency_lanes
            st.session_state.collisions      = frame_out.get("collisions", [])

            # Stats
            s = st.session_state.stats
            s["total_frames"]   += 1
            if is_new_crash:
                s["total_collisions"] += 1
            if frame_out.get("emergency_lane"):
                s["total_emergency"] += 1

            for v in frame_out["vehicles"]:
                vid = v.get("id")
                if vid is None:
                    continue

                if vid not in s["seen_vehicle_ids"]:
                    s["seen_vehicle_ids"].add(vid)
                    s["total_vehicles"] += 1

                d = v.get("direction", "unknown")
                if d not in s["seen_in_direction"]:
                    s["seen_in_direction"][d] = set()
                if vid not in s["seen_in_direction"][d]:
                    s["seen_in_direction"][d].add(vid)
                    s["direction_counts"][d] = s["direction_counts"].get(d, 0) + 1

                lane = v.get("lane", "unknown")
                if lane not in s["seen_in_lane"]:
                    s["seen_in_lane"][lane] = set()
                if vid not in s["seen_in_lane"][lane]:
                    s["seen_in_lane"][lane].add(vid)
                    s["lane_totals"][lane] = s["lane_totals"].get(lane, 0) + 1

            st.session_state.frames_processed = frame_index

            # Event log
            ts = datetime.now().strftime("%H:%M:%S")
            if is_new_crash:
                msg = f"CRASH [{crash_report['severity'].upper()}] lane={crash_report['lane']} score={crash_report['score']}"
                print(f"[ALERT] {msg}")
                st.session_state.event_log.appendleft(
                    {"type": "crash", "ts": ts, "msg": msg}
                )
            for emg in frame_out.get("emergency_lane", []):
                msg = f"EMERGENCY vehicle → {emg}"
                print(f"[ALERT] {msg}")
                st.session_state.event_log.appendleft(
                    {"type": "emerg", "ts": ts, "msg": msg}
                )

    except Exception as e:
        st.session_state.event_log.appendleft({
            "type": "crash", "ts": datetime.now().strftime("%H:%M:%S"),
            "msg": f"Pipeline error: {e}",
        })
        traceback.print_exc()
    finally:
        st.session_state.running = False


# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR — configuration & controls
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # st.markdown("## 🚦 Traffic AI") # Removed as requested
    st.markdown("---")

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
    # Hardcoded detection parameters
    conf_thresh = 0.20
    imgsz       = 1280
    frame_skip  = st.number_input("Frame skip", min_value=1, max_value=30, value=3, step=1)
    use_clahe   = st.checkbox("CLAHE (night/low-light)", value=True)

    st.markdown("### AI Mode")
    use_dqn = st.checkbox("Enable DQN signal control", value=False)

    st.markdown("---")
    col1, col2 = st.columns(2)
    start_btn = col1.button("▶ Start", width='stretch',
                             disabled=st.session_state.running or not source_path)
    stop_btn  = col2.button("⏹ Stop",  width='stretch',
                             disabled=not st.session_state.running)

    if start_btn and source_path:
        # Update path history
        if source_path in st.session_state.path_history:
            st.session_state.path_history.remove(source_path)
        st.session_state.path_history.insert(0, source_path)
        st.session_state.path_history = st.session_state.path_history[:3] # Max 3

        _init_state()
        st.session_state.running   = True
        st.session_state.stop_flag = False
        cfg = dict(
            conf_thresh=conf_thresh, imgsz=imgsz, frame_skip=frame_skip,
            use_clahe=use_clahe, use_dqn=use_dqn,
        )
        t = threading.Thread(target=pipeline_thread,
                             args=(source_path, cfg), daemon=True)
        add_script_run_ctx(t)
        t.start()
        st.rerun()

    if stop_btn:
        st.session_state.stop_flag = True
        st.session_state.running   = False
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
@st.fragment(run_every=0.12)
def render_dashboard_ui():
    # ─── HEADER ─────────────────────────────────────────────────────────────
    status_dot   = "🟢" if st.session_state.running else ("🔴" if st.session_state.frames_processed > 0 else "⚪")
    status_label = "LIVE" if st.session_state.running else ("STOPPED" if st.session_state.frames_processed > 0 else "IDLE")

    st.markdown(f"""
    <div class="dash-header">
      <div>
        <p class="dash-title">🚦 Traffic Command Center</p>
      </div>
      <div style="margin-left:auto; text-align:right; font-family:'JetBrains Mono',monospace; font-size:0.85rem; color:#64748b;">
        {status_dot} {status_label} &nbsp;|&nbsp; FPS: {st.session_state.current_fps:.1f} &nbsp;|&nbsp;
        Frames: {st.session_state.frames_processed}
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ─── TOP TOOLBAR: Signal + Metrics (50/50) ───────────────────────────────
    t_col1, t_col2 = st.columns([1, 1])

    with t_col1:
        st.markdown('<p class="section-head" style="margin-top:0;">Signal State</p>', unsafe_allow_html=True)
        sig = st.session_state.signal_output
        if sig:
            pills = ""
            for lane in ["top_road", "bottom_road", "left_road", "right_road"]:
                short = lane.replace("_road", "").upper()
                if lane in sig.get("active_lanes", []):
                    pills += f'<span class="signal-pill sig-green"><span class="sig-dot"></span>{short}</span>'
                elif lane in sig.get("yellow_lanes", []):
                    pills += f'<span class="signal-pill sig-yellow"><span class="sig-dot"></span>{short}</span>'
                else:
                    pills += f'<span class="signal-pill sig-red"><span class="sig-dot"></span>{short}</span>'
            
            override = sig.get("override_reason", "")
            if override and "standard" not in override:
                pills += f'<span style="font-family:JetBrains Mono;font-size:0.6rem;color:#ea580c;margin-left:8px;vertical-align:middle;">⚠ {override.upper()}</span>'
            
            st.markdown(f'<div class="signal-row" style="justify-content:flex-start;">{pills}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="color:#94a3b8;font-size:0.75rem;margin-top:8px;">SIGNAL OFFLINE</div>', unsafe_allow_html=True)

    with t_col2:
        st.markdown('<p class="section-head" style="margin-top:0;">System Stats</p>', unsafe_allow_html=True)
        s = st.session_state.stats
        
        def mini_metric_card(val, label, accent="#0ea5e9"):
            return f"""<div class="metric-card compact" style="--accent:{accent}">
              <div class="metric-val">{val}</div>
              <div class="metric-label">{label}</div>
            </div>"""

        m_cols = st.columns(4)
        m_cols[0].markdown(mini_metric_card(s["total_frames"], "FRAMES"), unsafe_allow_html=True)
        m_cols[1].markdown(mini_metric_card(s["total_vehicles"], "VEHICLES", "#6366f1"), unsafe_allow_html=True)
        m_cols[2].markdown(mini_metric_card(s["total_collisions"], "CRASHES", "#ef4444"), unsafe_allow_html=True)
        m_cols[3].markdown(mini_metric_card(s["total_emergency"], "EMERGENCY", "#f97316"), unsafe_allow_html=True)

    st.markdown('<hr style="margin: 15px 0; border: 0; border-top: 1px solid #e2e8f0;">', unsafe_allow_html=True)

    # ─── MAIN CONTENT LAYOUT ────────────────────────────────────────────────
    main_col, side_col = st.columns([2.2, 1])

    # ─── LEFT: LIVE FEED & LOGS ──
    with main_col:
        st.markdown('<p class="section-head" style="margin-top:0;">Live Intelligence Feed</p>', unsafe_allow_html=True)
        
        feed_container = st.container()
        with feed_container:
            if st.session_state.frame_rgb is not None:
                st.image(st.session_state.frame_rgb, width='stretch', channels="RGB")
            elif st.session_state.running:
                st.markdown(
                    '<div style="width:100%; aspect-ratio:16/9; background:#f1f5f9; border:1px dashed #cbd5e1; border-radius:12px;'
                    'display:flex; align-items:center; justify-content:center;'
                    'color:#0ea5e9; font-size:1rem; letter-spacing:4px; font-family:\'JetBrains Mono\',monospace;">'
                    'INITIALIZING SENSORS...</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div style="width:100%; aspect-ratio:16/9; background:#f1f5f9; border:1px dashed #cbd5e1; border-radius:12px;'
                    'display:flex; align-items:center; justify-content:center;'
                    'color:#94a3b8; font-size:1rem; letter-spacing:4px;">SYSTEM OFFLINE</div>',
                    unsafe_allow_html=True
                )

        # Event log at the bottom of main feed
        st.markdown('<p class="section-head">Real-time Event Log</p>', unsafe_allow_html=True)
        log_html = '<div style="max-height:350px;overflow-y:auto; background:#ffffff; border:1px solid #e2e8f0; border-radius:8px; padding:4px;">'
        if not st.session_state.event_log:
            log_html += '<div style="color:#94a3b8;font-size:0.72rem;text-align:center;padding:20px;">No events recorded yet.</div>'
        else:
            for ev in list(st.session_state.event_log)[:25]:
                css_cls = ev["type"]
                icon    = "💥" if ev["type"] == "crash" else ("🚨" if ev["type"] == "emerg" else "·")
                log_html += f'<div class="log-entry {css_cls}">[{ev["ts"]}] {icon} {ev["msg"]}</div>'
        log_html += '</div>'
        st.markdown(log_html, unsafe_allow_html=True)

    # ─── RIGHT: ANALYTICS & ALERTS ──
    with side_col:
        # Lane Density
        st.markdown('<p class="section-head" style="margin-top:0;">Live Lane Density</p>', unsafe_allow_html=True)
        lc = st.session_state.lane_counts
        lanes_display = ["top_road", "bottom_road", "left_road", "right_road"]
        bar_colors    = ["#eab308", "#22c55e", "#2563eb", "#ef4444"]
        
        n_cols = st.columns(2)
        for i, lane in enumerate(lanes_display):
            val = lc.get(lane, 0)
            n_cols[i % 2].markdown(f"""
                <div class="lane-stat-card" style="margin-bottom:8px;">
                    <div style="color:{bar_colors[i]}; font-size:1.6rem; font-family:'JetBrains Mono'; font-weight:700; line-height:1;">{val}</div>
                    <div style="color:#64748b; font-size:0.6rem; letter-spacing:1px; margin-top:4px;">{lane.replace('_road','').upper()}</div>
                </div>
            """, unsafe_allow_html=True)

        # Historical Totals
        st.markdown('<p class="section-head">Historical Lane Totals</p>', unsafe_allow_html=True)
        lt = st.session_state.stats.get("lane_totals", {})
        h_cols = st.columns(2)
        for i, lane in enumerate(lanes_display):
            val = lt.get(lane, 0)
            h_cols[i % 2].markdown(f"""
                <div class="lane-stat-card" style="margin-bottom:8px;">
                    <div style="color:{bar_colors[i]}; font-size:1.4rem; font-family:'JetBrains Mono'; font-weight:700; line-height:1;">{val}</div>
                    <div style="color:#64748b; font-size:0.55rem; letter-spacing:1px; margin-top:4px;">{lane.replace('_road','').upper()} TOTAL</div>
                </div>
            """, unsafe_allow_html=True)

        # Direction Distribution
        st.markdown('<p class="section-head">Direction Distribution</p>', unsafe_allow_html=True)
        dirs = st.session_state.stats["direction_counts"]
        if dirs:
            for d_label, d_val in list(dirs.items())[:6]:
                st.markdown(f"""
                    <div style="margin-bottom:6px; padding:4px 10px; background:#ffffff; border:1px solid #e2e8f0; border-left:3px solid #6366f1; border-radius:4px;">
                        <span style="color:#64748b; font-size:0.7rem; text-transform:uppercase;">{d_label}:</span>
                        <span style="color:#0f172a; font-family:'JetBrains Mono'; font-weight:700; float:right;">{d_val}</span>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown('<div style="color:#94a3b8;font-size:0.7rem;text-align:center;padding:10px;">Waiting for data…</div>', unsafe_allow_html=True)

        # Alerts section
        st.markdown('<p class="section-head">Dispatch Alerts</p>', unsafe_allow_html=True)
        # Emergency
        show_emerg = st.session_state.persisted_emerg and (time.time() - st.session_state.last_emerg_time < 10)
        if show_emerg:
            for lane in st.session_state.persisted_emerg:
                st.markdown(f'<div class="emergency-banner">🚨 EMERGENCY VEHICLE → <b>{lane.replace("_"," ").upper()}</b></div>', unsafe_allow_html=True)
        
        # Crash
        cr = st.session_state.persisted_crash
        show_crash = cr and (time.time() - st.session_state.last_crash_time < 10)
        if show_crash:
            st.markdown(f'<div class="alert-banner">💥 CRASH: <b>{cr["lane"].upper()}</b> | SEV: <b>{cr["severity"].upper()}</b></div>', unsafe_allow_html=True)
        
        if not show_emerg and not show_crash:
            st.markdown('<div style="color:#166534;background:#dcfce7;border:1px solid #b91c1c00;border-radius:8px;padding:10px;font-size:0.7rem;text-align:center;">✓ SYSTEM CLEAR</div>', unsafe_allow_html=True)


# ─── EXECUTE DASHBOARD ──────────────────────────────────────────────────────
render_dashboard_ui()
