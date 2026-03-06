"""Tank system: 36 rim tanks + 3 manifolds + hybrid plumbing.

Encapsulates constraint enforcement, water conservation repair,
and tank diagnostics. The actual tank ODE rates are computed inside
the dynamics model (rigid_body.py) as part of the derivative function;
this class handles the post-step constraint layer.
"""

from __future__ import annotations

import numpy as np

from habitat_sim.config import TankConfig
from habitat_sim.simulation.state import SimState


class TankSystem:
    """Manages the hybrid manifold tank system.

    Responsibilities:
        - Post-step constraint enforcement (clip tanks, fix conservation)
        - Diagnostics (fill levels, imbalance metrics)
        - Compute correction authority (what CM offset can be achieved)
    """

    def __init__(self, config: TankConfig):
        self.config = config
        self.n_per_station = config.n_tanks_per_station  # 12
        self.n_stations = config.n_stations              # 3
        self.n_tanks = config.n_tanks_total              # 36

    def enforce_constraints(self, state: SimState) -> None:
        """Hard-clip tanks to [0, max] and fix conservation drift.

        Should be called after every RK4 step.
        """
        tc = self.config

        # Clip tank masses
        np.clip(state.tank_masses, 0.0, tc.tank_capacity,
                out=state.tank_masses)

        # Clip manifold masses (no upper limit — manifold is flexible)
        np.clip(state.manifold_masses, 0.0, None,
                out=state.manifold_masses)

        # Fix total water conservation
        total = state.total_water()
        drift = total - tc.total_water_mass

        if abs(drift) > 1e-15:
            # Distribute correction to manifolds first (they're the buffers)
            m_sum = state.manifold_masses.sum()
            if m_sum > 1e-15:
                state.manifold_masses[:] -= drift * state.manifold_masses / m_sum
            else:
                # All water is in tanks — distribute there
                t_sum = state.tank_masses.sum()
                if t_sum > 1e-15:
                    state.tank_masses[:] -= drift * state.tank_masses / t_sum

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def station_fill_fractions(self, state: SimState) -> np.ndarray:
        """(3,) average fill fraction per station, in [0, 1]."""
        tank_2d = state.tank_masses_2d                # (3, 12)
        return tank_2d.mean(axis=1) / self.config.tank_capacity

    def angular_imbalance_per_station(
        self,
        state: SimState,
        tank_positions: np.ndarray,
    ) -> np.ndarray:
        """(3, 2) CM offset contribution [x, y] from tanks at each station.

        Returns body-frame (x, y) offset per station — useful for
        diagnosing which station is most imbalanced.
        """
        tank_2d = state.tank_masses_2d                # (3, 12)
        pos_2d = tank_positions.reshape(self.n_stations, self.n_per_station, 3)

        offsets = np.zeros((self.n_stations, 2))
        for j in range(self.n_stations):
            total_m = tank_2d[j].sum()
            if total_m > 1e-15:
                offsets[j, 0] = np.dot(tank_2d[j], pos_2d[j, :, 0]) / total_m
                offsets[j, 1] = np.dot(tank_2d[j], pos_2d[j, :, 1]) / total_m
        return offsets

    def compute_tank_cm_offset(
        self,
        state: SimState,
        tank_positions: np.ndarray,
        manifold_positions: np.ndarray,
    ) -> np.ndarray:
        """(3,) CM offset of all water relative to geometric centre."""
        total_water = state.total_water()
        if total_water < 1e-15:
            return np.zeros(3)

        moment = (np.dot(state.tank_masses, tank_positions)
                  + np.dot(state.manifold_masses, manifold_positions))
        return moment / total_water


# ---------------------------------------------------------------------------
# Utility: compute the target tank distribution to cancel a given CM offset
# ---------------------------------------------------------------------------

def compute_correction_target(
    desired_cm_x: float,
    desired_cm_y: float,
    tank_positions: np.ndarray,
    total_water: float,
    tank_capacity: float,
) -> np.ndarray:
    """Compute tank masses that place the water CM at (desired_cm_x, desired_cm_y, 0).

    Uses a least-squares approach: find the distribution that achieves
    the target CM while minimising deviation from uniform fill.

    This is not used by the RL agent (which learns its own strategy),
    but is useful for testing that the tank layout has sufficient authority.

    Args:
        desired_cm_x, desired_cm_y: target CM in body frame (m).
        tank_positions: (36, 3) body-frame tank positions.
        total_water: total water mass (kg).
        tank_capacity: per-tank max (kg).

    Returns:
        (36,) target tank masses summing to total_water,
        clipped to [0, tank_capacity].
    """
    n = len(tank_positions)
    uniform = total_water / n

    # We want:  Σ m_i * x_i = total_water * desired_cm_x
    #           Σ m_i * y_i = total_water * desired_cm_y
    #           Σ m_i = total_water
    # Minimise ||m - uniform||² subject to above.
    #
    # Lagrange multipliers or simple projection:
    # δm_i = λ_x * x_i + λ_y * y_i  (perturbation from uniform)
    #
    # Σ δm_i * x_i = total_water * desired_cm_x - uniform * Σ x_i
    # Σ δm_i * y_i = total_water * desired_cm_y - uniform * Σ y_i

    x = tank_positions[:, 0]
    y = tank_positions[:, 1]

    target_mx = total_water * desired_cm_x - uniform * x.sum()
    target_my = total_water * desired_cm_y - uniform * y.sum()

    # [Σ x_i², Σ x_i*y_i] [λ_x]   [target_mx]
    # [Σ x_i*y_i, Σ y_i²] [λ_y] = [target_my]
    A = np.array([
        [np.dot(x, x), np.dot(x, y)],
        [np.dot(x, y), np.dot(y, y)],
    ])
    b = np.array([target_mx, target_my])

    try:
        lam = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.full(n, uniform)

    masses = uniform + lam[0] * x + lam[1] * y
    np.clip(masses, 0.0, tank_capacity, out=masses)

    # Re-normalise to total_water
    masses *= total_water / masses.sum()

    return masses
