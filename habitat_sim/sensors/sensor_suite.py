"""Sensor suite: combines all sensor models into a single observation builder.

Observation vector layout (75 dimensions for 36-sector cylinder):
    [0:36]   strain gauge floor forces (N) — one per sector
    [36:72]  tank fill levels (36 tanks)
    [72:75]  manifold levels (3 manifolds)

Sector count and tank/manifold counts vary with habitat geometry; the indices
above assume the default 12-angular × 3-axial cylinder configuration.
"""

from __future__ import annotations

import numpy as np

from habitat_sim.config import SensorConfig
from habitat_sim.sensors.strain_gauge import StrainGaugeArray


class SensorSuite:
    """Combines all sensors and builds the RL observation vector."""

    def __init__(
        self,
        config: SensorConfig,
        sector_positions: np.ndarray,
        n_sectors: int = 36,
        n_tanks: int = 36,
        n_manifolds: int = 3,
        seed: int = 42,
    ):
        self.config = config
        self.n_sectors = n_sectors
        self.n_tanks = n_tanks
        self.n_manifolds = n_manifolds

        self.strain_gauges = StrainGaugeArray(
            sector_positions=sector_positions,
            noise_std=config.strain_gauge_noise_std,
        )

        self.rng = np.random.default_rng(seed)

        self.n_obs = n_sectors + n_tanks + n_manifolds

    def observe(
        self,
        omega: np.ndarray,
        d_omega: np.ndarray,
        sector_masses: np.ndarray,
        tank_masses: np.ndarray,
        manifold_masses: np.ndarray,
    ) -> np.ndarray:
        """Build the full observation vector.

        Args:
            omega:           (3,) body-frame angular velocity.
            d_omega:         (3,) body-frame angular acceleration.
            sector_masses:   (n_sectors,) true crew/cargo masses.
            tank_masses:     (n_tanks,) current tank fill levels.
            manifold_masses: (n_manifolds,) current manifold levels.

        Returns:
            (n_obs,) observation vector.
        """
        obs = np.empty(self.n_obs)

        # Strain gauge force readings
        obs[:self.n_sectors] = self.strain_gauges.measure(
            omega, d_omega, sector_masses, self.rng
        )

        # Tank fill levels (known exactly — agent's own actuators)
        i = self.n_sectors
        obs[i:i + self.n_tanks] = tank_masses

        # Manifold levels (known exactly)
        i += self.n_tanks
        obs[i:i + self.n_manifolds] = manifold_masses

        return obs

    def reset(self, seed: int | None = None) -> None:
        """Reset sensor state for a new episode."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    @property
    def observation_dimension(self) -> int:
        return self.n_obs
