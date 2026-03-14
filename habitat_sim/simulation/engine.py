"""Simulation engine: ties all components together.

Phase 3: adds sensor suite and observation building.
"""

from __future__ import annotations

import numpy as np

from habitat_sim.config import ExperimentConfig
from habitat_sim.geometry.cylinder import create_geometry
from habitat_sim.dynamics.rigid_body import RigidBodyDynamics
from habitat_sim.dynamics.base import DynamicsModel
from habitat_sim.actuators.motor import SpinMotor
from habitat_sim.actuators.tank_system import TankSystem
from habitat_sim.disturbances.scenario import Scenario, build_scenario
from habitat_sim.sensors.sensor_suite import SensorSuite
from habitat_sim.simulation.state import SimState
from habitat_sim.simulation.monitors import ConservationMonitor
from habitat_sim.core.integrator import rk4_step
from habitat_sim.core.inertia import compute_inertia_tensor, compute_cm_offset


def create_dynamics(level: int) -> DynamicsModel:
    """Factory for dynamics model by complexity level."""
    if level == 1:
        return RigidBodyDynamics()
    else:
        raise ValueError(f"Dynamics level {level} not implemented yet.")


class SimulationEngine:
    """Advances the simulation forward in time.

    Usage (Gymnasium-style):
        engine = SimulationEngine(config)
        obs, info = engine.step(action)

    Usage (manual sector masses, for testing):
        engine = SimulationEngine(config)
        obs, info = engine.step(action, sector_masses_override=masses)
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config

        # Build components
        self.geometry = create_geometry(config.habitat)
        self.dynamics = RigidBodyDynamics(config.tanks)
        self.motor = SpinMotor(config.motor)
        self.tank_system = TankSystem(config.tanks)
        self.monitor = ConservationMonitor(
            n_tanks=config.tanks.n_tanks_total,
            n_manifolds=config.tanks.n_stations,
        )
        self.state = SimState(config)
        self.t = 0.0

        # Build disturbance scenario
        self.scenario = build_scenario(
            config.disturbances, n_sectors=config.sectors.n_total)

        # Precompute static geometry data
        self.sector_positions = self.geometry.compute_sector_positions(
            config.sectors)
        self.tank_positions = self.geometry.compute_tank_positions(
            config.tanks)
        self.manifold_positions = self.geometry.compute_manifold_positions(
            config.tanks)
        self.structural_inertia = self.geometry.compute_structural_inertia()
        self.structural_mass = self.geometry.structural_mass()

        # Bundle into dict for the dynamics model
        self.precomputed = {
            "structural_inertia": self.structural_inertia,
            "structural_mass": self.structural_mass,
            "sector_positions": self.sector_positions,
            "tank_positions": self.tank_positions,
            "manifold_positions": self.manifold_positions,
            "tank_config": config.tanks,
        }

        # Build sensor suite (strain gauges use the same sector positions
        # that are already precomputed for the dynamics model)
        self.sensors = SensorSuite(
            config=config.sensors,
            sector_positions=self.sector_positions,
            n_sectors=config.sectors.n_total,
            n_tanks=config.tanks.n_tanks_total,
            n_manifolds=config.tanks.n_stations,
            seed=config.seed,
        )

        # Track state for observations
        self._last_sector_masses = np.zeros(config.sectors.n_total)
        self._last_d_omega = np.zeros(3)
        self._last_action = np.zeros(config.tanks.n_tanks_total)

    @property
    def observation_dimension(self) -> int:
        return self.sensors.observation_dimension

    @property
    def action_dimension(self) -> int:
        return self.config.tanks.n_tanks_total

    def step(
        self,
        action: np.ndarray,
        sector_masses_override: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Advance by one control interval (control_dt).

        Args:
            action: (36,) normalised valve commands in [-1, +1].
            sector_masses_override: if provided, bypass scenario.

        Returns:
            (observation, info) tuple.
        """
        dt = self.config.simulation.dt
        n_sub = self.config.simulation.n_substeps
        self._last_action = action

        for _ in range(n_sub):
            # Get sector masses
            if sector_masses_override is not None:
                sector_masses = sector_masses_override
            else:
                sector_masses = self.scenario.get_sector_masses(self.t)
            self._last_sector_masses = sector_masses

            motor_torque = self.motor.get_torque(self.t)

            # RK4 integration step
            self.state.x = rk4_step(
                self.dynamics.compute_derivatives,
                self.t,
                self.state.x,
                dt,
                sector_masses,
                action,
                motor_torque,
                self.precomputed,
            )

            # Post-step corrections
            q = self.state.quaternion
            q_norm = np.linalg.norm(q)
            if q_norm > 1e-15:
                q[:] = q / q_norm

            self.tank_system.enforce_constraints(self.state)

            # Conservation check
            self.monitor.check(
                self.t, self.state.x, self.precomputed,
                sector_masses, motor_torque,
            )

            self.t += dt

        # Compute dω/dt for the strain gauge Euler-acceleration term
        self._last_d_omega = self._compute_d_omega(sector_masses, action)

        # Build observation
        obs = self.sensors.observe(
            omega=self.state.omega,
            d_omega=self._last_d_omega,
            sector_masses=self._last_sector_masses,
            tank_masses=self.state.tank_masses,
            manifold_masses=self.state.manifold_masses,
        )

        info = self.monitor.get_info()
        info["d_omega"] = self._last_d_omega.copy()
        return obs, info

    def _compute_d_omega(
        self,
        sector_masses: np.ndarray,
        action: np.ndarray,
    ) -> np.ndarray:
        """Compute current angular acceleration from Euler's equation.

        Used by the strain gauge model after each control step.
        """
        omega = self.state.omega
        tank_masses = self.state.tank_masses
        manifold_masses = self.state.manifold_masses

        I = compute_inertia_tensor(
            self.structural_inertia,
            self.sector_positions, sector_masses,
            self.tank_positions, tank_masses,
            self.manifold_positions, manifold_masses,
        )

        motor_torque = self.motor.get_torque(self.t)
        tau_ext = np.array([0.0, 0.0, motor_torque])

        Iw = I @ omega
        gyroscopic = np.cross(omega, Iw)

        I_inv = np.linalg.inv(I)
        d_omega = I_inv @ (tau_ext - gyroscopic)
        return d_omega

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def step_no_control(
        self,
        sector_masses: np.ndarray | None = None,
    ) -> dict:
        """Step with zero valve commands.  Returns info only.

        Backward-compatible with Phase 1/2 tests.
        """
        action = np.zeros(self.config.tanks.n_tanks_total)
        _, info = self.step(action, sector_masses_override=sector_masses)
        return info

    def get_initial_observation(self) -> np.ndarray:
        """Build observation from current state without stepping."""
        sector_masses = self.scenario.get_sector_masses(self.t)
        d_omega = self._compute_d_omega(
            sector_masses,
            np.zeros(self.config.tanks.n_tanks_total),
        )
        return self.sensors.observe(
            omega=self.state.omega,
            d_omega=d_omega,
            sector_masses=sector_masses,
            tank_masses=self.state.tank_masses,
            manifold_masses=self.state.manifold_masses,
        )

    def get_nutation_angle(self) -> float:
        """Angle between body z-axis and inertial Z-axis (degrees)."""
        from habitat_sim.core.quaternion import quat_to_rotation_matrix
        R = quat_to_rotation_matrix(self.state.quaternion)
        z_body_in_inertial = R @ np.array([0.0, 0.0, 1.0])
        cos_theta = np.clip(z_body_in_inertial[2], -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_theta)))

    def get_cm_offset(self) -> np.ndarray:
        """(3,) current CM offset from geometric centre (m)."""
        return compute_cm_offset(
            self.structural_mass,
            self.sector_positions, self._last_sector_masses,
            self.tank_positions, self.state.tank_masses,
            self.manifold_positions, self.state.manifold_masses,
        )

    def get_cm_offset_magnitude(self) -> float:
        return float(np.linalg.norm(self.get_cm_offset()))

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reset engine state for a new episode.  Returns initial observation."""
        self.state = SimState(self.config)
        self.t = 0.0
        self.monitor.reset()
        self._last_sector_masses = np.zeros(self.config.sectors.n_total)
        self._last_d_omega = np.zeros(3)
        self._last_action = np.zeros(self.config.tanks.n_tanks_total)

        if seed is not None:
            self.sensors.reset(seed)

        return self.get_initial_observation()
