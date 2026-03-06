"""Sensor suite: combines all sensor models into a single observation builder.

Observation vector layout (93 dimensions for 6 accelerometers):
    [0:18]   accelerometer readings (6 sensors × 3 axes)
    [18:54]  sector mass estimates (36 sectors)
    [54:90]  tank fill levels (36 tanks)
    [90:93]  manifold levels (3 manifolds)
"""

from __future__ import annotations

import numpy as np

from habitat_sim.config import ExperimentConfig, SensorConfig
from habitat_sim.sensors.accelerometer import AccelerometerArray
from habitat_sim.sensors.mass_tracker import MassTracker


class SensorSuite:
    """Combines all sensors and builds the RL observation vector."""

    def __init__(
        self,
        config: SensorConfig,
        accelerometer_positions: np.ndarray,
        n_sectors: int = 36,
        n_tanks: int = 36,
        n_manifolds: int = 3,
        seed: int = 42,
    ):
        self.config = config
        self.n_accels = config.n_accelerometers
        self.n_sectors = n_sectors
        self.n_tanks = n_tanks
        self.n_manifolds = n_manifolds

        # Build sensor objects
        self.accelerometers = AccelerometerArray(
            positions=accelerometer_positions,
            noise_std=config.accelerometer_noise_std,
        )

        self.mass_tracker = MassTracker(
            n_sectors=n_sectors,
            noise_std=config.mass_tracker_noise_std,
            update_rate=config.mass_tracker_update_rate,
        )

        # RNG for sensor noise
        self.rng = np.random.default_rng(seed)

        # Observation dimensions
        self.n_accel_obs = 3 * self.n_accels
        self.n_obs = self.n_accel_obs + n_sectors + n_tanks + n_manifolds

    def observe(
        self,
        t: float,
        omega: np.ndarray,
        d_omega: np.ndarray,
        sector_masses: np.ndarray,
        tank_masses: np.ndarray,
        manifold_masses: np.ndarray,
    ) -> np.ndarray:
        """Build the full observation vector.

        Args:
            t:               current simulation time.
            omega:           (3,) body-frame angular velocity.
            d_omega:         (3,) body-frame angular acceleration.
            sector_masses:   (36,) true crew/cargo masses.
            tank_masses:     (36,) current tank fill levels.
            manifold_masses: (3,) current manifold levels.

        Returns:
            (n_obs,) observation vector.
        """
        obs = np.empty(self.n_obs)

        # Accelerometer readings
        accel_readings = self.accelerometers.measure_all_vectorised(
            omega, d_omega, self.rng
        )
        obs[:self.n_accel_obs] = accel_readings

        # Sector mass estimates (noisy, rate-limited)
        mass_est = self.mass_tracker.estimate(t, sector_masses, self.rng)
        i = self.n_accel_obs
        obs[i:i + self.n_sectors] = mass_est

        # Tank fill levels (known exactly -- these are the agent's own actuators)
        i += self.n_sectors
        obs[i:i + self.n_tanks] = tank_masses

        # Manifold levels (known exactly)
        i += self.n_tanks
        obs[i:i + self.n_manifolds] = manifold_masses

        return obs

    def reset(self, seed: int | None = None) -> None:
        """Reset sensor state for a new episode."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.mass_tracker.reset()

    @property
    def observation_dimension(self) -> int:
        return self.n_obs
