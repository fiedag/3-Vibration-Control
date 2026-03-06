"""Abstract dynamics model interface.

This is the central abstraction that allows swapping Level 1 (rigid body)
for Level 2 (+ 2 flex modes) or Level 3 (+ 20 flex modes) without changing
the integrator, engine, or environment code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class DynamicsModel(ABC):
    """Interface for computing state derivatives."""

    @abstractmethod
    def compute_derivatives(
        self,
        t: float,
        state: np.ndarray,
        sector_masses: np.ndarray,
        tank_valve_commands: np.ndarray,
        motor_torque: float,
        precomputed: dict,
    ) -> np.ndarray:
        """Compute dx/dt for the full state vector.

        Args:
            t:  current simulation time (s).
            state: flat state vector.
            sector_masses: (N_sectors,) crew/cargo masses at this instant.
            tank_valve_commands: (N_tanks,) normalised commands in [-1, +1].
            motor_torque: scalar torque about z_B (N·m).
            precomputed: dict of geometry-derived constants:
                - "structural_inertia": (3,3)
                - "structural_mass": float
                - "sector_positions": (N_sectors, 3)
                - "tank_positions": (N_tanks, 3)
                - "manifold_positions": (N_stations, 3)
                - "tank_config": TankConfig

        Returns:
            dx/dt as flat array with same length as state.
        """
        ...

    @abstractmethod
    def state_dimension(self) -> int:
        """Total number of state variables for this dynamics level."""
        ...
