"""Gymnasium environment for the rotating space habitat.

Wraps SimulationEngine to provide the standard Gymnasium API
for reinforcement learning with SAC.

Observation space (75-dim for 36-sector cylinder):
    [0:36]   strain gauge floor forces (N) — one per sector
    [36:72]  tank fill levels (36)
    [72:75]  manifold levels (3)

Action space (36-dim):
    Normalised valve commands in [-1, +1], one per tank.

Reward:
    Weighted combination of vibration suppression, pump energy,
    command smoothness, and reserve balance.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

try:
    import gymnasium
    from gymnasium import spaces
    HAS_GYMNASIUM = True
except ImportError:
    HAS_GYMNASIUM = False

from habitat_sim.config import ExperimentConfig, reference_config
from habitat_sim.simulation.engine import SimulationEngine


def _require_gymnasium():
    if not HAS_GYMNASIUM:
        raise ImportError(
            "gymnasium is required for HabitatEnv. "
            "Install with: pip install gymnasium"
        )


class HabitatEnv(gymnasium.Env):
    """Gymnasium environment for the habitat simulation.

    Provides the standard gymnasium.Env interface for SAC training.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: ExperimentConfig | None = None,
        reward_weights: dict | None = None,
        render_mode: str | None = None,
    ):
        _require_gymnasium()
        super().__init__()

        self.config = config or reference_config()
        self.render_mode = render_mode

        # Build engine to get dimensions
        self.engine = SimulationEngine(self.config)

        n_obs = self.engine.observation_dimension
        n_act = self.engine.action_dimension

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(n_obs,), dtype=np.float64,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(n_act,), dtype=np.float64,
        )

        # Reward weights
        rw = reward_weights or {}
        self._w_vibration = rw.get("vibration", 1.0)
        self._w_energy = rw.get("energy", 0.01)
        self._w_smooth = rw.get("smooth", 0.005)
        self._w_reserve = rw.get("reserve", 0.001)

        # Track previous action for smoothness penalty
        self._prev_action = np.zeros(n_act)

        # Control step counter
        self._step_count = 0
        self._max_steps = int(
            self.config.simulation.duration / self.config.simulation.control_dt
        )

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset environment for a new episode."""
        super().reset(seed=seed)

        if seed is not None:
            self.config.seed = seed
        else:
            seed = self.config.seed

        self.engine = SimulationEngine(self.config)
        obs = self.engine.reset(seed=seed)

        self._prev_action = np.zeros(self.engine.action_dimension)
        self._step_count = 0

        return obs, {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Take one control step.

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        # Clip action to valid range
        action = np.clip(action, -1.0, 1.0)

        obs, info = self.engine.step(action)

        reward = self._compute_reward(action, info)

        self._prev_action = action.copy()
        self._step_count += 1

        terminated = self._step_count >= self._max_steps
        truncated = False

        # Add useful metrics to info
        info["step_count"] = self._step_count
        info["nutation_angle_deg"] = self.engine.get_nutation_angle()

        return obs, reward, terminated, truncated, info

    def _compute_reward(self, action: np.ndarray, info: dict) -> float:
        """Compute weighted reward.

        Components:
            - vibration:  penalise transverse angular velocity (wobble)
            - energy:     penalise total pump activity
            - smooth:     penalise jerky commands
            - reserve:    prefer balanced tank fill levels
        """
        omega = self.engine.state.omega

        # --- Vibration: penalise ω_x and ω_y (wobble) ---
        wobble = np.sqrt(omega[0]**2 + omega[1]**2)
        r_vibration = -wobble

        # --- Energy: penalise total valve activity ---
        r_energy = -np.sum(np.abs(action))

        # --- Smoothness: penalise change from previous action ---
        r_smooth = -np.sum(np.abs(action - self._prev_action))

        # --- Reserve: penalise deviation from uniform fill ---
        tank_masses = self.engine.state.tank_masses
        uniform = self.config.tanks.total_water_mass / (
            self.config.tanks.n_tanks_total + self.config.tanks.n_stations
        )
        r_reserve = -np.sum((tank_masses - uniform)**2) / len(tank_masses)

        reward = (
            self._w_vibration * r_vibration
            + self._w_energy * r_energy
            + self._w_smooth * r_smooth
            + self._w_reserve * r_reserve
        )

        return float(reward)

    def render(self):
        """Placeholder for rendering."""
        pass

    def close(self):
        """Clean up."""
        pass


# ---------------------------------------------------------------------------
# Registration helper (optional, for gymnasium.make)
# ---------------------------------------------------------------------------

def register_env():
    """Register HabitatEnv with gymnasium if available."""
    if HAS_GYMNASIUM:
        gymnasium.register(
            id="HabitatSim-v0",
            entry_point="habitat_sim.environment.habitat_env:HabitatEnv",
        )
