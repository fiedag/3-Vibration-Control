"""Conservation law monitors and diagnostic checks.

Run at every physics step to detect integration errors or bugs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from habitat_sim.core.quaternion import quat_to_rotation_matrix
from habitat_sim.core.inertia import compute_inertia_tensor, compute_cm_offset


@dataclass
class ConservationRecord:
    """Snapshot of conservation quantities at one time step."""
    t: float
    h_inertial: np.ndarray          # (3,) angular momentum in inertial frame
    kinetic_energy: float
    quaternion_norm: float
    total_water: float
    cm_offset: np.ndarray           # (3,) CM offset from geometric centre


class ConservationMonitor:
    """Tracks conservation quantities and flags violations."""

    def __init__(self, h_tol: float = 1e-6, q_tol: float = 1e-10,
                 water_tol: float = 1e-10,
                 n_tanks: int = 36, n_manifolds: int = 3):
        self.h_tol = h_tol
        self.q_tol = q_tol
        self.water_tol = water_tol
        self._tank_start = 7
        self._tank_end = 7 + n_tanks
        self._manifold_end = self._tank_end + n_manifolds

        self._initial_h: np.ndarray | None = None
        self._initial_water: float | None = None
        self._history: list[ConservationRecord] = []
        self._violations: list[str] = []

    def check(
        self,
        t: float,
        state_x: np.ndarray,
        precomputed: dict,
        sector_masses: np.ndarray,
        motor_torque: float,
    ) -> ConservationRecord:
        """Compute and record conservation quantities.

        Args:
            t: current time.
            state_x: flat state vector.
            precomputed: dict of geometry constants.
            sector_masses: (36,) current crew/cargo masses.
            motor_torque: current motor torque (to know if τ_ext = 0).

        Returns:
            ConservationRecord for this step.
        """
        q = state_x[0:4]
        omega = state_x[4:7]
        tank_masses = state_x[self._tank_start:self._tank_end]
        manifold_masses = state_x[self._tank_end:self._manifold_end]

        # Inertia tensor
        I = compute_inertia_tensor(
            precomputed["structural_inertia"],
            precomputed["sector_positions"], sector_masses,
            precomputed["tank_positions"], tank_masses,
            precomputed["manifold_positions"], manifold_masses,
        )

        # Angular momentum in body frame, then rotate to inertial
        h_body = I @ omega
        R = quat_to_rotation_matrix(q)
        h_inertial = R @ h_body

        # Kinetic energy
        ke = 0.5 * omega @ h_body

        # Quaternion norm
        q_norm = np.linalg.norm(q)

        # Total water
        total_water = float(tank_masses.sum() + manifold_masses.sum())

        # CM offset
        cm = compute_cm_offset(
            precomputed["structural_mass"],
            precomputed["sector_positions"], sector_masses,
            precomputed["tank_positions"], tank_masses,
            precomputed["manifold_positions"], manifold_masses,
        )

        record = ConservationRecord(
            t=t,
            h_inertial=h_inertial.copy(),
            kinetic_energy=ke,
            quaternion_norm=q_norm,
            total_water=total_water,
            cm_offset=cm.copy(),
        )

        # --- Check violations ---
        if self._initial_h is None:
            self._initial_h = h_inertial.copy()
            self._initial_water = total_water

        # Angular momentum (only meaningful when no external torque)
        if abs(motor_torque) < 1e-12:
            h_err = np.linalg.norm(h_inertial - self._initial_h)
            h_ref = np.linalg.norm(self._initial_h)
            if h_ref > 1e-12 and h_err / h_ref > self.h_tol:
                self._violations.append(
                    f"t={t:.4f}: H conservation error "
                    f"{h_err/h_ref:.2e} > tol {self.h_tol:.2e}"
                )

        # Quaternion norm
        if abs(q_norm - 1.0) > self.q_tol:
            self._violations.append(
                f"t={t:.4f}: |q| = {q_norm:.12f} (drift {abs(q_norm-1.0):.2e})"
            )

        # Water conservation
        if self._initial_water is not None:
            w_err = abs(total_water - self._initial_water)
            if w_err > self.water_tol:
                self._violations.append(
                    f"t={t:.4f}: water drift {w_err:.2e} kg"
                )

        self._history.append(record)
        return record

    def get_info(self) -> dict:
        """Return summary info for the current episode so far."""
        if not self._history:
            return {}
        last = self._history[-1]
        return {
            "h_inertial": last.h_inertial,
            "kinetic_energy": last.kinetic_energy,
            "cm_offset": last.cm_offset,
            "cm_offset_mag": float(np.linalg.norm(last.cm_offset)),
            "quaternion_norm": last.quaternion_norm,
            "total_water": last.total_water,
            "n_violations": len(self._violations),
            "violations": self._violations[-5:],  # last 5
        }

    def reset(self) -> None:
        """Clear history for a new episode."""
        self._initial_h = None
        self._initial_water = None
        self._history.clear()
        self._violations.clear()
