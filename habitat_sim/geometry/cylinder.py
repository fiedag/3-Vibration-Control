"""Cylinder geometry: thin-walled shell with two end plates."""

from __future__ import annotations

import numpy as np

from habitat_sim.config import HabitatConfig
from habitat_sim.geometry.base import HabitatGeometry


class CylinderGeometry(HabitatGeometry):
    """Closed cylinder (shell + two end plates)."""

    def structural_mass(self) -> float:
        c = self.config
        m_shell = c.wall_density * 2.0 * np.pi * c.radius * c.wall_thickness * c.length
        m_end = c.end_plate_density * np.pi * c.radius**2 * c.end_plate_thickness
        return m_shell + 2.0 * m_end

    def compute_structural_inertia(self) -> np.ndarray:
        """Inertia tensor about geometric centre for thin-walled cylinder.

        Formulae (body frame, z_B = symmetry axis):
            Shell:
                I_xx = I_yy = m_s * (R²/2 + L²/12)
                I_zz = m_s * R²
            Each end plate (at ±L/2):
                I_xx_plate = m_e * R²/4  (about plate centroid)
                Parallel-axis shift: + m_e * (L/2)²
            Off-diagonal = 0 by symmetry.
        """
        c = self.config
        R, L = c.radius, c.length
        t_w, rho_w = c.wall_thickness, c.wall_density
        t_e, rho_e = c.end_plate_thickness, c.end_plate_density

        m_shell = rho_w * 2.0 * np.pi * R * t_w * L
        m_end = rho_e * np.pi * R**2 * t_e  # mass of ONE end plate

        # Shell contributions
        I_xx_shell = m_shell * (R**2 / 2.0 + L**2 / 12.0)
        I_zz_shell = m_shell * R**2

        # End plate contributions (two plates, each at ±L/2)
        I_xx_end = 2.0 * m_end * (R**2 / 4.0 + (L / 2.0)**2)
        I_zz_end = 2.0 * m_end * R**2 / 2.0

        I = np.zeros((3, 3))
        I[0, 0] = I_xx_shell + I_xx_end
        I[1, 1] = I[0, 0]                    # axisymmetric
        I[2, 2] = I_zz_shell + I_zz_end
        return I


class RingGeometry(HabitatGeometry):
    """Open cylinder (no end plates)."""

    def structural_mass(self) -> float:
        c = self.config
        return c.wall_density * 2.0 * np.pi * c.radius * c.wall_thickness * c.length

    def compute_structural_inertia(self) -> np.ndarray:
        c = self.config
        R, L = c.radius, c.length
        m_shell = self.structural_mass()

        I = np.zeros((3, 3))
        I[0, 0] = m_shell * (R**2 / 2.0 + L**2 / 12.0)
        I[1, 1] = I[0, 0]
        I[2, 2] = m_shell * R**2
        return I


# -----------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------

def create_geometry(config: HabitatConfig) -> HabitatGeometry:
    """Instantiate the correct geometry class from config."""
    from habitat_sim.geometry.toroid import ToroidGeometry
    _MAP = {
        "cylinder": CylinderGeometry,
        "ring": RingGeometry,
        "toroid": ToroidGeometry,
    }
    cls = _MAP.get(config.shape)
    if cls is None:
        raise ValueError(f"Unknown habitat shape: {config.shape!r}")
    return cls(config)
