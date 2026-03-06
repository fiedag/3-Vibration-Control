"""Level 1: Rigid body dynamics with variable inertia.

Implements Euler's equations for a rotating body whose inertia tensor
changes due to moving crew/cargo and water redistribution in rim tanks.

State vector (46 elements):
    [0:4]   quaternion  (q0, q1, q2, q3)
    [4:7]   angular velocity  (ω_x, ω_y, ω_z) in body frame
    [7:43]  tank masses  (36 tanks, row-major by station then angular)
    [43:46] manifold masses  (3 manifolds)
"""

from __future__ import annotations

import numpy as np

from habitat_sim.dynamics.base import DynamicsModel
from habitat_sim.core.quaternion import quat_derivative
from habitat_sim.core.inertia import compute_inertia_tensor


class RigidBodyDynamics(DynamicsModel):
    """Level 1 dynamics: Euler equations + tank/manifold ODEs."""

    # Default state vector layout (for 36 tanks + 3 manifolds)
    _Q_SLICE = slice(0, 4)
    _W_SLICE = slice(4, 7)
    _TANK_START = 7
    _MANIFOLD_OFFSET = 43  # = 7 + 36
    _N_STATE = 46

    def __init__(self, tank_config=None) -> None:
        """Optionally accept a TankConfig to support non-default tank counts."""
        if tank_config is not None:
            n_tanks = tank_config.n_tanks_total
            n_manifolds = tank_config.n_stations
        else:
            n_tanks = 36
            n_manifolds = 3
        self._tank_start = 7
        self._manifold_offset = self._tank_start + n_tanks
        self._n_state = self._manifold_offset + n_manifolds
        self._n_tanks = n_tanks
        self._n_manifolds = n_manifolds

    def state_dimension(self) -> int:
        return self._n_state

    def compute_derivatives(
        self,
        t: float,
        state: np.ndarray,
        sector_masses: np.ndarray,
        tank_valve_commands: np.ndarray,
        motor_torque: float,
        precomputed: dict,
    ) -> np.ndarray:
        # ---------------------------------------------------------------
        # Unpack state
        # ---------------------------------------------------------------
        q = state[self._Q_SLICE]           # (4,) quaternion
        omega = state[self._W_SLICE]       # (3,) angular velocity
        tank_end = self._manifold_offset
        tank_masses = state[self._tank_start:tank_end]
        manifold_masses = state[tank_end:tank_end + self._n_manifolds]

        # ---------------------------------------------------------------
        # Unpack precomputed geometry constants
        # ---------------------------------------------------------------
        structural_inertia = precomputed["structural_inertia"]
        sector_positions   = precomputed["sector_positions"]
        tank_positions     = precomputed["tank_positions"]
        manifold_positions = precomputed["manifold_positions"]
        tank_config        = precomputed["tank_config"]

        # ---------------------------------------------------------------
        # Inertia tensor
        # ---------------------------------------------------------------
        I = compute_inertia_tensor(
            structural_inertia,
            sector_positions, sector_masses,
            tank_positions, tank_masses,
            manifold_positions, manifold_masses,
        )

        # ---------------------------------------------------------------
        # Tank dynamics (circumferential flow + axial equalisation)
        # ---------------------------------------------------------------
        n_per_station = tank_config.n_tanks_per_station  # 12
        n_stations = tank_config.n_stations              # 3
        q_circ_max = tank_config.q_circ_max
        q_axial_max = tank_config.q_axial_max
        k_axial = tank_config.k_axial
        tank_cap = tank_config.tank_capacity

        # Circumferential flow: valve command × max rate
        flow_rates = tank_valve_commands * q_circ_max     # (36,)

        # Clip: can't drain an empty tank or overfill a full tank
        flow_rates = np.where(
            (flow_rates > 0) & (tank_masses >= tank_cap),
            0.0, flow_rates
        )
        flow_rates = np.where(
            (flow_rates < 0) & (tank_masses <= 0.0),
            0.0, flow_rates
        )

        # Net flow out of each manifold  (positive flow_rate = fill tank = drain manifold)
        flow_2d = flow_rates.reshape(n_stations, n_per_station)
        manifold_drain = -flow_2d.sum(axis=1)              # (3,)

        # Throttle if manifold would go negative
        for j in range(n_stations):
            if manifold_drain[j] < 0 and manifold_masses[j] <= 0.0:
                # This manifold is empty; can't drain more
                # Zero out all fill commands for this station
                flow_rates[j * n_per_station:(j + 1) * n_per_station] = np.where(
                    flow_rates[j * n_per_station:(j + 1) * n_per_station] > 0,
                    0.0,
                    flow_rates[j * n_per_station:(j + 1) * n_per_station],
                )
                manifold_drain[j] = -flow_rates[
                    j * n_per_station:(j + 1) * n_per_station
                ].sum()

        # Axial equalisation: proportional control driving manifolds to their mean
        mean_level = manifold_masses.mean()
        axial_error = mean_level - manifold_masses         # positive = under-filled
        axial_flow = np.clip(axial_error * k_axial,
                             -q_axial_max, q_axial_max)
        axial_flow -= axial_flow.mean()                    # enforce conservation

        d_tanks = flow_rates
        d_manifolds = manifold_drain + axial_flow

        # ---------------------------------------------------------------
        # Compute dI/dt from tank flow rates
        # ---------------------------------------------------------------
        # dI/dt = Σ (dm_k/dt) * (|r_k|² I₃ - r_k r_kᵀ)
        # Only tank masses are changing within a physics step;
        # sector masses are treated as piecewise constant.
        d_masses = np.concatenate([d_tanks, d_manifolds])
        d_positions = np.concatenate([tank_positions, manifold_positions])

        r_sq = np.sum(d_positions ** 2, axis=1)
        outer = d_positions[:, :, None] * d_positions[:, None, :]
        dI_dt = np.sum(
            d_masses[:, None, None] * (r_sq[:, None, None] * np.eye(3) - outer),
            axis=0,
        )

        # ---------------------------------------------------------------
        # Euler's equation:  dω/dt = I⁻¹ [τ_ext − dI/dt·ω − ω×(Iω)]
        # ---------------------------------------------------------------
        tau_ext = np.array([0.0, 0.0, motor_torque])

        Iw = I @ omega
        gyroscopic = np.cross(omega, Iw)
        inertia_rate = dI_dt @ omega

        I_inv = np.linalg.inv(I)
        d_omega = I_inv @ (tau_ext - inertia_rate - gyroscopic)

        # ---------------------------------------------------------------
        # Quaternion kinematics:  dq/dt = ½ Ω(ω) q
        # ---------------------------------------------------------------
        d_quat = quat_derivative(q, omega)

        # ---------------------------------------------------------------
        # Assemble derivative vector
        # ---------------------------------------------------------------
        dx = np.empty(self._n_state)
        dx[self._Q_SLICE] = d_quat
        dx[self._W_SLICE] = d_omega
        dx[self._tank_start:self._manifold_offset] = d_tanks
        dx[self._manifold_offset:self._manifold_offset + self._n_manifolds] = d_manifolds

        return dx
