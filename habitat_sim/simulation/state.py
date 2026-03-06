"""Simulation state: flat NumPy array with named views.

The state vector layout for Level 1 (46 states):
    [0:4]    quaternion (q0, q1, q2, q3)
    [4:7]    angular velocity (ω_x, ω_y, ω_z) body frame
    [7:43]   tank masses (36 tanks, row-major: station×angular)
    [43:46]  manifold masses (3 manifolds)
"""

from __future__ import annotations

import numpy as np

from habitat_sim.config import ExperimentConfig


class SimState:
    """Manages the full state vector as a flat numpy array with named views."""

    def __init__(self, config: ExperimentConfig):
        self.n_rigid = 7
        self.n_tanks = config.tanks.n_tanks_total        # 36
        self.n_manifolds = config.tanks.n_stations       # 3
        self.n_total = self.n_rigid + self.n_tanks + self.n_manifolds  # 46

        self._tank_end = self.n_rigid + self.n_tanks     # 43
        self._manifold_end = self._tank_end + self.n_manifolds  # 46

        self.x = np.zeros(self.n_total)
        self.x[0] = 1.0   # quaternion identity: [1, 0, 0, 0]

        # Initialise tank water
        self._init_water(config)

    def _init_water(self, config: ExperimentConfig) -> None:
        """Set initial water distribution across tanks and manifolds."""
        tc = config.tanks
        if tc.initial_distribution == "uniform":
            # Split water evenly: most in tanks, remainder in manifolds
            water_per_tank = tc.total_water_mass / (tc.n_tanks_total + tc.n_stations)
            self.tank_masses[:] = water_per_tank
            self.manifold_masses[:] = water_per_tank
            # Correct for rounding to hit exact total
            drift = (self.tank_masses.sum() + self.manifold_masses.sum()
                     - tc.total_water_mass)
            self.manifold_masses[:] -= drift / tc.n_stations
        else:
            # Default: all water in tanks, manifolds empty
            self.tank_masses[:] = tc.total_water_mass / tc.n_tanks_total
            self.manifold_masses[:] = 0.0

    # ------------------------------------------------------------------
    # Named views (zero-copy slices into self.x)
    # ------------------------------------------------------------------

    @property
    def quaternion(self) -> np.ndarray:
        """(4,) quaternion [w, x, y, z]."""
        return self.x[0:4]

    @property
    def omega(self) -> np.ndarray:
        """(3,) angular velocity [ω_x, ω_y, ω_z] in body frame."""
        return self.x[4:7]

    @property
    def tank_masses(self) -> np.ndarray:
        """(36,) flat view of all tank masses."""
        return self.x[self.n_rigid:self._tank_end]

    @property
    def tank_masses_2d(self) -> np.ndarray:
        """(3, 12) view: [station, angular_index]."""
        return self.x[self.n_rigid:self._tank_end].reshape(
            -1, self.n_tanks // 3  # n_stations, n_per_station
        )

    @property
    def manifold_masses(self) -> np.ndarray:
        """(3,) manifold water masses."""
        return self.x[self._tank_end:self._manifold_end]

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def total_water(self) -> float:
        """Sum of all water in tanks + manifolds (conservation check)."""
        return float(self.tank_masses.sum() + self.manifold_masses.sum())

    def copy(self) -> "SimState":
        """Deep copy."""
        clone = object.__new__(SimState)
        clone.n_rigid = self.n_rigid
        clone.n_tanks = self.n_tanks
        clone.n_manifolds = self.n_manifolds
        clone.n_total = self.n_total
        clone._tank_end = self._tank_end
        clone._manifold_end = self._manifold_end
        clone.x = self.x.copy()
        return clone
