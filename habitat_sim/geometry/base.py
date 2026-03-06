"""Abstract base class for habitat geometry."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from habitat_sim.config import HabitatConfig, SectorConfig, TankConfig


class HabitatGeometry(ABC):
    """Computes static geometry-dependent quantities.

    Subclasses implement the analytical inertia, sector positions,
    and tank positions for each shape (cylinder, ring, toroid).
    """

    def __init__(self, config: HabitatConfig):
        self.config = config

    @abstractmethod
    def compute_structural_inertia(self) -> np.ndarray:
        """Return (3,3) inertia tensor of the empty structure about
        the geometric centre in body-frame coordinates."""
        ...

    @abstractmethod
    def structural_mass(self) -> float:
        """Total mass of the empty structure (shell + end plates etc.)."""
        ...

    def compute_sector_positions(self, sc: SectorConfig) -> np.ndarray:
        """Return (N_total, 3) body-frame positions of sector centroids.

        Default implementation works for cylinder and ring — place
        sectors at radius R on the inner wall.  Toroid overrides this.
        """
        R = self.config.radius
        L = self.config.length
        n_ang = sc.n_angular
        n_ax = sc.n_axial

        d_theta = 2.0 * np.pi / n_ang
        d_z = L / n_ax

        positions = np.zeros((n_ang * n_ax, 3))
        idx = 0
        for j in range(n_ax):
            z_j = -L / 2.0 + (j + 0.5) * d_z
            for i in range(n_ang):
                theta_i = (i + 0.5) * d_theta
                positions[idx] = [R * np.cos(theta_i),
                                  R * np.sin(theta_i),
                                  z_j]
                idx += 1
        return positions

    def compute_tank_positions(self, tc: TankConfig) -> np.ndarray:
        """Return (N_tanks_total, 3) body-frame positions of rim tanks.

        Tanks are co-located angularly with sectors, at the rim radius,
        at each axial station.
        """
        R = self.config.radius
        L = self.config.length
        n_ang = tc.n_tanks_per_station
        n_st = tc.n_stations

        d_theta = 2.0 * np.pi / n_ang
        d_z = L / n_st

        positions = np.zeros((n_ang * n_st, 3))
        idx = 0
        for j in range(n_st):
            z_j = -L / 2.0 + (j + 0.5) * d_z
            for i in range(n_ang):
                theta_i = (i + 0.5) * d_theta
                positions[idx] = [R * np.cos(theta_i),
                                  R * np.sin(theta_i),
                                  z_j]
                idx += 1
        return positions

    def compute_manifold_positions(self, tc: TankConfig) -> np.ndarray:
        """Return (N_stations, 3) body-frame positions of manifolds.

        Manifolds sit on-axis at each axial station: [0, 0, z_j].
        They contribute no angular imbalance.
        """
        L = self.config.length
        n_st = tc.n_stations
        d_z = L / n_st

        positions = np.zeros((n_st, 3))
        for j in range(n_st):
            positions[j, 2] = -L / 2.0 + (j + 0.5) * d_z
        return positions

    def compute_default_accelerometer_positions(
        self, n_accels: int = 6
    ) -> np.ndarray:
        """Place accelerometers: 2 per axial station, opposed at 0° and 180°.

        Returns (n_accels, 3).  Assumes n_accels is even and
        n_accels // 2 <= n_stations (typically 6 accels, 3 stations).
        """
        R = self.config.radius
        L = self.config.length
        n_stations = n_accels // 2
        d_z = L / n_stations

        positions = np.zeros((n_accels, 3))
        for j in range(n_stations):
            z_j = -L / 2.0 + (j + 0.5) * d_z
            positions[2 * j]     = [ R, 0.0, z_j]   # 0°
            positions[2 * j + 1] = [-R, 0.0, z_j]   # 180°
        return positions
