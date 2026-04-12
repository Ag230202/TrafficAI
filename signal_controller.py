"""
signal_controller.py
--------------------
Adaptive traffic signal control for a 4-way intersection.

Reads frame-by-frame traffic analysis (lane_counts, emergency_lane, collisions)
and outputs phase timings with three priority layers:
  1. Emergency preemption (ambulances/fire trucks get priority)
  2. Collision red override (force red on lanes with active crashes)
  3. Density-proportional green time (scale signal duration by vehicle count)
  4. Anti-starvation guard (every lane gets min_green per cycle)

State management:
  - Tracks current phase and elapsed time within that phase
  - Re-computes green duration every frame based on current input
  - Transitions phases when elapsed >= computed duration

Usage:
  controller = SignalController(SIGNAL_CONFIG)
  for frame_output in run_pipeline(...):
      signal_phase = controller.update(
          frame_output["lane_counts"],
          frame_output["emergency_lane"],
          frame_output["collisions"],
          frame_id=frame_output["frame_id"]
      )
      # signal_phase now contains phase_id, green_duration, active_lane, etc.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import json


# ─────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────

SIGNAL_CONFIG = {
    # ── Timing (seconds) ────────────────────────────────────────
    "cycle_duration":           60,      # Total time for one full phase rotation
    "base_green_duration":      20,      # Starting green for each phase
    "min_green_duration":       8,       # Never go below this (pedestrians, etc.)
    "max_green_duration":       50,      # Never exceed this (prevent too-long waits)
    "yellow_duration":          4,       # Time to show yellow before switching
    "all_red_duration":         1,       # Safety gap between phases
    
    # ── Emergency settings ──────────────────────────────────────
    "emergency_duration":       25,      # Green time for emergency vehicles
    "emergency_preemption":     True,    # Enable emergency priority
    
    # ── Density scaling ─────────────────────────────────────────
    "density_scaling_factor":   0.8,     # How aggressively to scale by demand
                                          # Higher = more responsive to density
    "enable_adaptive":          True,    # Scale green time by lane counts
    "use_dqn":                  False,   # Phase 3 offline reinforcement learning
    
    # ── Anti-starvation ────────────────────────────────────
    "enable_anti_starvation":   True,    # Guarantee min_green per lane per cycle
    "max_wait_cycles":           3,       # Promote a lane after being skipped this many times
    
    # ── Collision handling ──────────────────────────────────────
    "collision_red_timeout":    5,       # Frames to keep lane red after collision
    "enable_collision_override": True,   # Hard-stop traffic into collision zone
    
    # ── Phase definitions (North-South / East-West quad) ────────
    "phases": [
        {
            "id": 0,
            "name": "North-South (Top + Bottom)",
            "lanes": ["top_road", "bottom_road"],
            "default_green": 20,
        },
        {
            "id": 1,
            "name": "East (Right)",
            "lanes": ["right_road"],
            "default_green": 15,
        },
        {
            "id": 2,
            "name": "North-South (Top + Bottom)",
            "lanes": ["top_road", "bottom_road"],
            "default_green": 20,
        },
        {
            "id": 3,
            "name": "West (Left)",
            "lanes": ["left_road"],
            "default_green": 15,
        },
    ],
    
    # ── Debug ───────────────────────────────────────────────────
    "debug_mode":               False,   # Print signal state each frame
    "log_file":                 "signal_log.txt",
}


# ─────────────────────────────────────────────────────────────────
#  STATE AND OUTPUT MODELS
# ─────────────────────────────────────────────────────────────────

@dataclass
class SignalControllerState:
    """Internal state for phase tracking within a single pipeline run."""
    current_phase_id: int = 0
    elapsed_in_phase: float = 0.0          # Seconds (fractional)
    phase_green_duration: int = 20          # Current computed green time
    phase_start_frame: int = 0
    is_yellow_mode: bool = False            # Transitioning to next phase
    yellow_elapsed: float = 0.0
    all_red_elapsed: float = 0.0
    collision_cooldown: Dict[str, int] = field(default_factory=dict)  # lane -> frames_remaining
    last_frame_id: int = -1
    wait_cycle_count: Dict[str, int] = field(default_factory=dict)   # lane -> skipped cycles
    target_phase_id: Optional[int] = None
    current_override: Optional[str] = None
    target_phase_id: Optional[int] = None
    current_override: Optional[str] = None
    
    def reset_phase(self, phase_id: int, frame_id: int):
        """Reset counters when transitioning to a new phase."""
        self.current_phase_id = phase_id
        self.elapsed_in_phase = 0.0
        self.phase_start_frame = frame_id
        self.is_yellow_mode = False
        self.yellow_elapsed = 0.0
        self.all_red_elapsed = 0.0


@dataclass
class SignalPhaseOutput:
    """Output dict for one frame's signal state."""
    phase_id: int
    phase_name: str
    active_lanes: List[str]           # Lanes with GREEN
    green_duration: int                # Seconds of green
    red_lanes: List[str]               # Lanes with RED
    yellow_lanes: List[str]            # Lanes in transition (YELLOW)
    next_phase_id: int                 # Precomputed next phase
    elapsed_in_phase: float            # How long we've been in current phase
    time_until_next: float             # Seconds until next phase
    is_yellow_mode: bool               # Currently in yellow transition
    override_reason: Optional[str]     # Why we deviated (emergency, collision, etc.)
    confidence: float                  # 0.0–1.0: certainty of this timing
    metadata: Dict = field(default_factory=dict)  # Debug info
    
    def to_dict(self) -> dict:
        """Convert to plain dict for JSON serialization."""
        return {
            "phase_id": self.phase_id,
            "phase_name": self.phase_name,
            "active_lanes": self.active_lanes,
            "green_duration": self.green_duration,
            "red_lanes": self.red_lanes,
            "yellow_lanes": self.yellow_lanes,
            "next_phase_id": self.next_phase_id,
            "elapsed_in_phase": round(self.elapsed_in_phase, 2),
            "time_until_next": round(self.time_until_next, 2),
            "is_yellow_mode": self.is_yellow_mode,
            "override_reason": self.override_reason,
            "confidence": round(self.confidence, 2),
            "metadata": self.metadata,
        }


# ─────────────────────────────────────────────────────────────────
#  SIGNAL CONTROLLER
# ─────────────────────────────────────────────────────────────────

class SignalController:
    """
    Adaptive traffic signal controller for a 4-way intersection.
    
    Inputs (per frame):
      - lane_counts: dict of {lane_name: vehicle_count}
      - emergency_lane: list of lane names with emergency vehicles
      - collisions: list of collision dicts with 'lane' key
    
    Outputs (per frame):
      - SignalPhaseOutput: complete signal state for current frame
    
    Assumptions:
      - Intersection has 4 lanes: top_road, bottom_road, left_road, right_road
      - Phases are pre-defined (configurable)
      - Time is measured per frame (assuming constant frame rate)
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or SIGNAL_CONFIG
        self.state = SignalControllerState()
        
        # Parse phases
        self.phases = {
            p["id"]: p for p in self.config.get("phases", [])
        }
        self.phase_order = [p["id"] for p in self.config.get("phases", [])]
        
        # For frame-to-time conversion: assume 30 fps (override if different)
        self.fps = 30
        self.frame_skip = 3  # Default; can be updated
        self.seconds_per_frame = (self.frame_skip / self.fps)
        
        self.dqn_agent = None
        if self.config.get("use_dqn", False):
            try:
                from dqn_agent import DQNAgent
                self.dqn_agent = DQNAgent()
            except ImportError as e:
                print(f"[SignalController] WARNING: Could not import DQNAgent: {e}")
    
    def set_frame_rate(self, fps: int, frame_skip: int):
        """Set frame rate info for accurate time tracking."""
        self.fps = fps
        self.frame_skip = frame_skip
        self.seconds_per_frame = (frame_skip / fps)
    
    def update(
        self,
        lane_counts: Dict[str, int],
        emergency_lane: List[str],
        collisions: List[Dict],
        frame_id: int = 0,
    ) -> SignalPhaseOutput:
        """
        Update controller state and compute signal phase for current frame.
        
        Args:
            lane_counts: {lane_name: count} from Phase 1
            emergency_lane: list of lane names with emergency vehicles
            collisions: list of collision dicts with 'lane' key
            frame_id: current frame number
        
        Returns:
            SignalPhaseOutput with all signal state info
        """
        # ── Time tracking ────────────────────────────────────────
        if self.state.last_frame_id < 0:
            # First frame
            self.state.last_frame_id = frame_id
            frame_delta = 0.0
        else:
            # Assume sequential frames; if gap detected, assume missed frames
            frame_delta = (frame_id - self.state.last_frame_id) * self.seconds_per_frame
        
        self.state.last_frame_id = frame_id
        
        # ── Decay collision cooldown ─────────────────────────────
        for lane in list(self.state.collision_cooldown.keys()):
            self.state.collision_cooldown[lane] -= 1
            if self.state.collision_cooldown[lane] <= 0:
                del self.state.collision_cooldown[lane]
        
        # ── Register new collisions ──────────────────────────────
        if self.config.get("enable_collision_override", True):
            for collision in collisions:
                lane = collision.get("lane")
                if lane:
                    self.state.collision_cooldown[lane] = \
                        self.config.get("collision_red_timeout", 5)
        
        # ── Check for emergency preemption ───────────────────────
        if self.config.get("emergency_preemption", True) and emergency_lane:
            # Immediate phase switch to first emergency lane's phase
            output = self._handle_emergency_preemption(
                emergency_lane, lane_counts, frame_id
            )
            if self.config.get("debug_mode", False):
                print(f"[Signal] Frame {frame_id}: EMERGENCY PREEMPTION → {output.phase_name}")
            return output
        
        # ── Standard phase progression ───────────────────────────
        self.state.elapsed_in_phase += frame_delta
        
        # Determine intent (Rule-based vs DQN)
        min_green = self.config.get("min_green_duration", 8)
        self.state.current_override = None
        
        if self.config.get("use_dqn", False) and self.dqn_agent:
            # Phase 3: DQN Logic
            state_vector = self._build_dqn_state_vector(lane_counts)
            dqn_action = self.dqn_agent.predict(state_vector)
            
            # Priority 3 Check: Anti-starvation overrides DQN!
            starved_phase = self._check_anti_starvation_only()
            if starved_phase is not None:
                target_phase_id = starved_phase
                self.state.current_override = "anti_starvation"
            else:
                target_phase_id = dqn_action
                self.state.current_override = "dqn_agent"
                
            # Compute a dummy green duration for UI (force switch if needed)
            if target_phase_id != self.state.current_phase_id:
                green_duration = max(min_green, self.state.elapsed_in_phase)
            else:
                green_duration = self.config.get("max_green_duration", 50)
            self.state.phase_green_duration = green_duration
            
        else:
            # Phase 2: Compute green duration for current phase
            phase_def = self.phases[self.state.current_phase_id]
            green_duration = self._compute_adaptive_green_time(
                phase_def, lane_counts
            )
            self.state.phase_green_duration = green_duration
            target_phase_id = None
        
        # Check if we should transition to next phase
        transition_threshold = green_duration + self.config.get("yellow_duration", 4)
        
        if self.state.elapsed_in_phase >= transition_threshold and not self.state.is_yellow_mode:
            # Enter yellow mode
            self.state.is_yellow_mode = True
            self.state.yellow_elapsed = 0.0
            
            # Pre-calculate target for yellow transition
            if target_phase_id is not None:
                self.state.target_phase_id = target_phase_id
            else:
                self.state.target_phase_id = self._get_next_phase_id(self.state.current_phase_id)
        
        elif self.state.is_yellow_mode:
            # Track time in yellow
            self.state.yellow_elapsed += frame_delta
            yellow_dur = self.config.get("yellow_duration", 4)
            
            if self.state.yellow_elapsed >= yellow_dur:
                # Transition to next phase using predefined target
                next_phase_id = getattr(self.state, "target_phase_id", None)
                if next_phase_id is None:
                    next_phase_id = self._get_next_phase_id(self.state.current_phase_id)
                self.state.reset_phase(next_phase_id, frame_id)
                phase_def = self.phases[self.state.current_phase_id]
                self.state.phase_green_duration = self._compute_adaptive_green_time(phase_def, lane_counts)
        
        # ── Build output ────────────────────────────────────────
        phase_def = self.phases[self.state.current_phase_id]
        output = self._build_phase_output(
            phase_def, lane_counts, collisions, frame_id, override_reason=getattr(self.state, "current_override", None)
        )
        
        if self.config.get("debug_mode", False):
            print(f"[Signal] Frame {frame_id}: {output.phase_name} "
                  f"(elapsed={output.elapsed_in_phase:.1f}s / "
                  f"green={output.green_duration}s)")
        
        return output
    
    def _handle_emergency_preemption(
        self,
        emergency_lane: List[str],
        lane_counts: Dict[str, int],
        frame_id: int,
    ) -> SignalPhaseOutput:
        """
        Jump to the phase that serves the first emergency lane.
        """
        target_lane = emergency_lane[0]  # Prioritize first in list
        
        # Find phase that serves this lane
        target_phase_id = None
        for phase_id, phase_def in self.phases.items():
            if target_lane in phase_def.get("lanes", []):
                target_phase_id = phase_id
                break
        
        if target_phase_id is None:
            # Fallback to current phase if emergency lane not in any phase
            target_phase_id = self.state.current_phase_id
        
        # Force switch to emergency phase
        self.state.reset_phase(target_phase_id, frame_id)
        emergency_duration = self.config.get("emergency_duration", 25)
        self.state.phase_green_duration = emergency_duration
        
        phase_def = self.phases[target_phase_id]
        output = self._build_phase_output(
            phase_def, lane_counts, [], frame_id,
            override_reason="emergency_preemption",
            confidence=1.0,
            green_override=emergency_duration
        )
        return output
    
    def _compute_adaptive_green_time(
        self,
        phase_def: Dict,
        lane_counts: Dict[str, int]
    ) -> int:
        """
        Compute green duration for a phase based on lane demand (density).
        
        Density-proportional formula:
          green_time = base_duration * (lane_demand / total_vehicles) + min_green
        
        Where lane_demand = sum of vehicle counts in this phase's lanes.
        """
        if not self.config.get("enable_adaptive", True):
            # Return default green duration
            return phase_def.get("default_green", self.config.get("base_green_duration", 20))
        
        # Sum vehicle counts for lanes in this phase
        phase_lanes = phase_def.get("lanes", [])
        phase_demand = sum(lane_counts.get(lane, 0) for lane in phase_lanes)
        total_demand = sum(lane_counts.values())
        
        if total_demand == 0:
            # No vehicles — use min_green to allow pedestrians, etc.
            return self.config.get("min_green_duration", 8)
        
        # Density-proportional scaling
        scaling_factor = self.config.get("density_scaling_factor", 0.8)
        base_duration = self.config.get("cycle_duration", 60)
        demand_ratio = phase_demand / total_demand
        
        green_time = int(base_duration * scaling_factor * demand_ratio) \
                     + self.config.get("min_green_duration", 8)
        
        # Clamp to min/max
        min_green = self.config.get("min_green_duration", 8)
        max_green = self.config.get("max_green_duration", 50)
        green_time = max(min_green, min(max_green, green_time))
        
        return green_time
    


    def _build_dqn_state_vector(self, lane_counts: Dict[str, int]) -> List[float]:
        lanes = ["left_road", "bottom_road", "right_road", "top_road"]
        counts = [float(lane_counts.get(l, 0)) for l in lanes]
        waits = [float(self.state.wait_cycle_count.get(l, 0)) for l in lanes]
        return counts + waits + [float(self.state.current_phase_id), float(round(self.state.elapsed_in_phase, 2))]

    def _check_anti_starvation_only(self) -> Optional[int]:
        if not self.config.get("enable_anti_starvation", True):
            return None
        max_wait = self.config.get("max_wait_cycles", 3)
        best_starved_phase = None
        best_wait = 0
        for phase_id, phase_def in self.phases.items():
            if phase_id == self.state.current_phase_id:
                continue
            for lane in phase_def.get("lanes", []):
                wait = self.state.wait_cycle_count.get(lane, 0)
                if wait >= max_wait and wait > best_wait:
                    best_wait = wait
                    best_starved_phase = phase_id
        return best_starved_phase



    def _get_next_phase_id(self, current_phase_id: int) -> int:
        """
        Get the ID of the next phase in rotation.
        
        Anti-starvation guard (Priority 3):
          If any lane has been skipped for ≥ max_wait_cycles consecutive cycles,
          its phase is promoted to run next regardless of density turn order.
          The wait counter resets each time a lane gets a green phase.
        """
        try:
            current_idx  = self.phase_order.index(current_phase_id)
            default_next = self.phase_order[(current_idx + 1) % len(self.phase_order)]
        except (ValueError, IndexError):
            return 0

        if not self.config.get("enable_anti_starvation", True):
            return default_next

        max_wait = self.config.get("max_wait_cycles", 3)

        # Find the phase whose lanes have been waiting longest
        best_starved_phase = None
        best_wait          = 0

        for phase_id, phase_def in self.phases.items():
            if phase_id == current_phase_id:
                continue
            for lane in phase_def.get("lanes", []):
                wait = self.state.wait_cycle_count.get(lane, 0)
                if wait >= max_wait and wait > best_wait:
                    best_wait          = wait
                    best_starved_phase = phase_id

        # Update wait counters: increment for every skipped phase's lanes,
        # reset for the lanes that just got green (current phase)
        for phase_id, phase_def in self.phases.items():
            for lane in phase_def.get("lanes", []):
                if phase_id == current_phase_id:
                    self.state.wait_cycle_count[lane] = 0   # just served
                else:
                    self.state.wait_cycle_count[lane] = \
                        self.state.wait_cycle_count.get(lane, 0) + 1

        if best_starved_phase is not None:
            return best_starved_phase  # Anti-starvation promotion

        return default_next
    
    def _build_phase_output(
        self,
        phase_def: Dict,
        lane_counts: Dict[str, int],
        collisions: List[Dict],
        frame_id: int,
        override_reason: Optional[str] = None,
        confidence: float = 0.8,
        green_override: Optional[int] = None,
    ) -> SignalPhaseOutput:
        """
        Build complete output object for current phase.
        """
        phase_id = phase_def.get("id", 0)
        phase_name = phase_def.get("name", "Unknown Phase")
        phase_lanes = phase_def.get("lanes", [])
        
        # Use override if provided (e.g., emergency), else compute
        green_duration = green_override or self.state.phase_green_duration
        
        # Determine which lanes get what signal
        active_lanes = []
        red_lanes = []
        yellow_lanes = []
        
        # ── Collision red override ───────────────────────────────
        collision_red_lanes = set()
        if self.config.get("enable_collision_override", True):
            for lane in self.state.collision_cooldown.keys():
                collision_red_lanes.add(lane)
        
        # ── Apply collision override: force red on affected lanes ─
        for lane_name in lane_counts.keys():
            if lane_name == "unknown":
                continue
            
            if lane_name in collision_red_lanes:
                # Collision → force RED (don't care about phase)
                red_lanes.append(lane_name)
            elif lane_name in phase_lanes:
                # This phase's lane
                if self.state.is_yellow_mode:
                    yellow_lanes.append(lane_name)
                else:
                    active_lanes.append(lane_name)
            else:
                # Other phase's lane → RED
                red_lanes.append(lane_name)
        
        # Time remaining in current green phase
        time_until_next = max(0.0, green_duration - self.state.elapsed_in_phase)
        
        next_phase_id = self._get_next_phase_id(phase_id)
        
        # ── Build metadata for debug ────────────────────────────
        metadata = {
            "total_vehicles": sum(lane_counts.values()),
            "phase_demand": sum(
                lane_counts.get(lane, 0) for lane in phase_lanes
            ),
            "collision_cooldowns": dict(self.state.collision_cooldown),
        }
        
        # Set override reason if not already set
        if not override_reason:
            if collision_red_lanes:
                override_reason = "collision_red_override"
                confidence = 0.95
            else:
                override_reason = "standard_adaptive" if self.config.get("enable_adaptive") \
                                else "standard_fixed"
                confidence = 0.8
        
        return SignalPhaseOutput(
            phase_id=phase_id,
            phase_name=phase_name,
            active_lanes=active_lanes,
            green_duration=green_duration,
            red_lanes=red_lanes,
            yellow_lanes=yellow_lanes,
            next_phase_id=next_phase_id,
            elapsed_in_phase=self.state.elapsed_in_phase,
            time_until_next=time_until_next,
            is_yellow_mode=self.state.is_yellow_mode,
            override_reason=override_reason,
            confidence=confidence,
            metadata=metadata,
        )