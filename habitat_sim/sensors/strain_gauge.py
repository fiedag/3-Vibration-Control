"""Strain gauge sensor model for sector floor force measurement.

Each sector floor (the outer rim surface) has a strain gauge that measures
the compressive normal force from the mass occupying that sector.

In the rotating body frame the "artificial gravity" pushes occupants and
cargo radially outward against the floor.  The measured force is:

    F_i = m_i × (-r̂_i · (ω × (ω × r_i) + dω/dt × r_i)) + noise

where:
    r_i   — body-frame position of sector i centroid (m)
    r̂_i   — outward radial unit vector for sector i (xy-plane projection)
    m_i   — total crew/cargo mass in sector i (kg)
    ω     — body-frame angular velocity (rad/s)
    dω/dt — body-frame angular acceleration (rad/s²)

For steady spin (ω = [0, 0, ω_z]) the centripetal term gives:
    F_i ≈ m_i × ω_z² × R   (always positive; the artificial-gravity weight)

Wobble (non-zero ω_x, ω_y) creates a sinusoidal variation in F_i around
the ring, which gives the agent information about the current nutation state.
"""

from __future__ import annotations

import numpy as np


class StrainGaugeArray:
    """Array of strain gauges — one per sector floor.

    Produces a (n_sectors,) force observation in Newtons.
    """

    def __init__(
        self,
        sector_positions: np.ndarray,
        noise_std: float = 0.0,
    ):
        """
        Args:
            sector_positions: (n_sectors, 3) body-frame centroid positions.
            noise_std:        standard deviation of white Gaussian noise per
                              gauge (N).
        """
        self.positions = np.asarray(sector_positions, dtype=np.float64)
        self.n_sectors = len(self.positions)
        self.noise_std = noise_std

        # Outward radial unit vectors — project sector positions onto the
        # xy-plane (perpendicular to the spin axis z) and normalise.
        r_xy = self.positions.copy()
        r_xy[:, 2] = 0.0                          # zero the axial component
        norms = np.linalg.norm(r_xy, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)  # avoid divide-by-zero
        self._r_hat = r_xy / norms                # (n_sectors, 3)

    def measure(
        self,
        omega: np.ndarray,
        d_omega: np.ndarray,
        sector_masses: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Compute strain gauge readings for all sectors.

        Args:
            omega:        (3,) body-frame angular velocity (rad/s).
            d_omega:      (3,) body-frame angular acceleration (rad/s²).
            sector_masses:(n_sectors,) crew/cargo mass per sector (kg).
            rng:          random generator for noise.  None = no noise.

        Returns:
            (n_sectors,) force readings in Newtons (positive = compressive).
        """
        r = self.positions                        # (N, 3)

        # Centripetal acceleration at each sector: ω × (ω × r)
        omega_cross_r = np.cross(omega, r)        # (N, 3)
        centripetal   = np.cross(omega, omega_cross_r)  # (N, 3)

        # Euler acceleration: dω/dt × r
        euler = np.cross(d_omega, r)              # (N, 3)

        # Total inertial acceleration at each sector centroid
        a = centripetal + euler                   # (N, 3)

        # Component normal to the floor = -r̂ · a
        # (centripetal points inward, so -r̂·centripetal = ω²R > 0)
        radial_a = -np.einsum("ij,ij->i", self._r_hat, a)  # (N,)

        # Force = mass × normal acceleration
        forces = sector_masses * radial_a         # (N,)

        if self.noise_std > 0 and rng is not None:
            forces = forces + rng.normal(0.0, self.noise_std, size=self.n_sectors)

        return forces
