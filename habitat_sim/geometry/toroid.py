"""Toroid geometry: thin-walled toroidal shell."""

from __future__ import annotations

import numpy as np

from habitat_sim.config import HabitatConfig, SectorConfig, TankConfig
from habitat_sim.geometry.base import HabitatGeometry


class ToroidGeometry(HabitatGeometry):
    """Thin-walled toroidal shell.

    Uses config.radius (R, major radius) and config.minor_radius (r, minor radius).

    Inertia formulae for a thin-walled toroidal shell of total mass m:
        I_zz = m * (2*R^2 + 3*r^2) / 2   (spin axis through torus centre)
        I_xx = I_yy = m * (2*R^2 + 5*r^2) / 4

    Structural mass via thin-wall approximation:
        m = wall_density * 4 * pi^2 * R * r * wall_thickness
        (surface area of torus = 4*pi^2*R*r, thin-wall volume = area * thickness)

    Sectors are arranged at the major radius in the equatorial (z=0) plane;
    n_axial is ignored -- a toroid has no axial extent.
    """

    def structural_mass(self) -> float:
        c = self.config
        surface_area = 4.0 * np.pi**2 * c.radius * c.minor_radius
        return c.wall_density * surface_area * c.wall_thickness

    def compute_structural_inertia(self) -> np.ndarray:
        m = self.structural_mass()
        R = self.config.radius
        r = self.config.minor_radius

        I = np.zeros((3, 3))
        I[2, 2] = m * (2.0 * R**2 + 3.0 * r**2) / 2.0
        I[0, 0] = m * (2.0 * R**2 + 5.0 * r**2) / 4.0
        I[1, 1] = I[0, 0]
        return I

    def compute_sector_positions(self, sc: SectorConfig) -> np.ndarray:
        """Sectors at major radius in the equatorial plane.

        n_axial is ignored; returns (n_angular, 3) array.
        """
        R = self.config.radius
        n_ang = sc.n_angular
        d_theta = 2.0 * np.pi / n_ang

        positions = np.zeros((n_ang, 3))
        for i in range(n_ang):
            theta_i = (i + 0.5) * d_theta
            positions[i] = [R * np.cos(theta_i), R * np.sin(theta_i), 0.0]
        return positions

    def compute_tank_positions(self, tc: TankConfig) -> np.ndarray:
        """Tank positions at major radius in equatorial plane.

        n_stations is ignored for toroid; returns (n_tanks_per_station, 3).
        """
        R = self.config.radius
        n_ang = tc.n_tanks_per_station
        d_theta = 2.0 * np.pi / n_ang

        positions = np.zeros((n_ang, 3))
        for i in range(n_ang):
            theta_i = (i + 0.5) * d_theta
            positions[i] = [R * np.cos(theta_i), R * np.sin(theta_i), 0.0]
        return positions

    def compute_manifold_positions(self, tc: TankConfig) -> np.ndarray:
        """Single manifold at origin (toroid has one axial station)."""
        return np.zeros((1, 3))

    def compute_default_accelerometer_positions(
        self, n_accels: int = 6
    ) -> np.ndarray:
        """Two opposed sensors in equatorial plane at (R,0,0) and (-R,0,0).

        Returns (2, 3) regardless of n_accels.
        """
        R = self.config.radius
        return np.array([[R, 0.0, 0.0], [-R, 0.0, 0.0]])
