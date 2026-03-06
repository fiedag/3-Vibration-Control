"""Inertia tensor computation.

All tensors are computed about the geometric centre in the body frame.
Vectorised for performance — this is called at every RK4 sub-step.
"""

from __future__ import annotations

import numpy as np

_EYE3 = np.eye(3)


def point_mass_inertia(positions: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Compute inertia tensor contribution from an array of point masses.

    Args:
        positions: (N, 3) body-frame positions of point masses.
        masses:    (N,)   mass of each point.

    Returns:
        (3, 3) inertia tensor about the origin (geometric centre).

    Uses the standard formula for each point mass k:
        I_k = m_k * (|r_k|² I₃ - r_k r_kᵀ)
    """
    # |r_k|²  shape (N,)
    r_sq = np.sum(positions ** 2, axis=1)

    # r_k r_kᵀ  shape (N, 3, 3)
    outer = positions[:, :, None] * positions[:, None, :]

    # Weighted sum:  Σ m_k * (|r_k|² I₃ - r_k r_kᵀ)
    I = np.sum(
        masses[:, None, None] * (r_sq[:, None, None] * _EYE3 - outer),
        axis=0,
    )
    return I


def compute_inertia_tensor(
    structural_inertia: np.ndarray,
    sector_positions: np.ndarray,
    sector_masses: np.ndarray,
    tank_positions: np.ndarray,
    tank_masses: np.ndarray,
    manifold_positions: np.ndarray,
    manifold_masses: np.ndarray,
) -> np.ndarray:
    """Compute total inertia tensor about geometric centre.

    Args:
        structural_inertia:  (3,3) precomputed from geometry, constant.
        sector_positions:    (36, 3) precomputed, constant.
        sector_masses:       (36,) time-varying crew/cargo masses.
        tank_positions:      (36, 3) precomputed, constant.
        tank_masses:         (36,) time-varying from state vector.
        manifold_positions:  (3, 3) precomputed, constant (on-axis).
        manifold_masses:     (3,) time-varying from state vector.

    Returns:
        (3, 3) total inertia tensor.
    """
    I = structural_inertia.copy()

    # Batch all point masses together for a single vectorised pass
    all_positions = np.concatenate([sector_positions, tank_positions,
                                    manifold_positions])
    all_masses = np.concatenate([sector_masses, tank_masses, manifold_masses])

    I += point_mass_inertia(all_positions, all_masses)
    return I


def compute_cm_offset(
    structural_mass: float,
    sector_positions: np.ndarray,
    sector_masses: np.ndarray,
    tank_positions: np.ndarray,
    tank_masses: np.ndarray,
    manifold_positions: np.ndarray,
    manifold_masses: np.ndarray,
) -> np.ndarray:
    """Compute centre-of-mass offset from geometric centre.

    Structural CM is assumed at the origin (symmetric structure).

    Returns:
        (3,) CM position in body frame.
    """
    total_mass = (structural_mass
                  + sector_masses.sum()
                  + tank_masses.sum()
                  + manifold_masses.sum())

    if total_mass < 1e-15:
        return np.zeros(3)

    moment = (np.dot(sector_masses, sector_positions)
              + np.dot(tank_masses, tank_positions)
              + np.dot(manifold_masses, manifold_positions))
    # structural_mass contribution is zero (origin)

    return moment / total_mass
