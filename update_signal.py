import re
import os

path = r"d:\Traffic_AI\signal_controller.py"

with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Add config
content = content.replace(
    '"enable_adaptive":          True,    # Scale green time by lane counts',
    '"enable_adaptive":          True,    # Scale green time by lane counts\n    "use_dqn":                  False,   # Phase 3 offline reinforcement learning'
)

# 2. Add State fields
content = content.replace(
    'wait_cycle_count: Dict[str, int] = field(default_factory=dict)   # lane -> skipped cycles',
    'wait_cycle_count: Dict[str, int] = field(default_factory=dict)   # lane -> skipped cycles\n    target_phase_id: Optional[int] = None\n    current_override: Optional[str] = None'
)

# 3. Add DQNAgent initialization
content = content.replace(
    'self.seconds_per_frame = (self.frame_skip / self.fps)',
    'self.seconds_per_frame = (self.frame_skip / self.fps)\n        \n        self.dqn_agent = None\n        if self.config.get("use_dqn", False):\n            try:\n                from dqn_agent import DQNAgent\n                self.dqn_agent = DQNAgent()\n            except ImportError as e:\n                print(f"[SignalController] WARNING: Could not import DQNAgent: {e}")'
)

# 4. Replace update phase progression
old_update = """        # ── Standard phase progression ───────────────────────────
        self.state.elapsed_in_phase += frame_delta
        
        # Compute green duration for current phase
        phase_def = self.phases[self.state.current_phase_id]
        green_duration = self._compute_adaptive_green_time(
            phase_def, lane_counts
        )
        self.state.phase_green_duration = green_duration
        
        # Check if we should transition to next phase
        transition_threshold = green_duration + self.config.get("yellow_duration", 4)
        
        if self.state.elapsed_in_phase >= transition_threshold and not self.state.is_yellow_mode:
            # Enter yellow mode
            self.state.is_yellow_mode = True
            self.state.yellow_elapsed = 0.0
        
        elif self.state.is_yellow_mode:
            # Track time in yellow
            self.state.yellow_elapsed += frame_delta
            yellow_dur = self.config.get("yellow_duration", 4)
            
            if self.state.yellow_elapsed >= yellow_dur:
                # Transition to next phase
                next_phase_id = self._get_next_phase_id(self.state.current_phase_id)
                self.state.reset_phase(next_phase_id, frame_id)
                phase_def = self.phases[self.state.current_phase_id]
                green_duration = self._compute_adaptive_green_time(
                    phase_def, lane_counts
                )
                self.state.phase_green_duration = green_duration"""

new_update = """        # ── Standard phase progression ───────────────────────────
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
                self.state.phase_green_duration = self._compute_adaptive_green_time(phase_def, lane_counts)"""

content = content.replace(old_update, new_update)

# 5. Build output to include override reason
old_build_output = """        # ── Build output ────────────────────────────────────────
        output = self._build_phase_output(
            phase_def, lane_counts, collisions, frame_id
        )"""

new_build_output = """        # ── Build output ────────────────────────────────────────
        phase_def = self.phases[self.state.current_phase_id]
        output = self._build_phase_output(
            phase_def, lane_counts, collisions, frame_id, override_reason=getattr(self.state, "current_override", None)
        )"""

content = content.replace(old_build_output, new_build_output)

# 6. Add helper methods
helpers = """

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

    def _get_next_phase_id(self, current_phase_id: int) -> int:"""

content = content.replace("    def _get_next_phase_id(self, current_phase_id: int) -> int:", helpers)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)

print("[SUCCESS] signal_controller.py updated")
