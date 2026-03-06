"""Spin motor model with configurable torque profiles.

The motor applies torque about the body z-axis (symmetry axis)
for spin-up and spin-down.
"""

from __future__ import annotations

import numpy as np

from habitat_sim.config import MotorConfig


class SpinMotor:
    """Provides τ_motor(t) according to the configured profile."""

    def __init__(self, config: MotorConfig):
        self.config = config
        self._profile_fn = self._build_profile()

    def get_torque(self, t: float) -> float:
        """Return motor torque (N·m) at time t."""
        return self._profile_fn(t)

    # ------------------------------------------------------------------
    # Profile builders
    # ------------------------------------------------------------------

    def _build_profile(self):
        c = self.config
        name = c.profile.lower()

        if name == "constant":
            return self._constant
        elif name == "ramp":
            return self._ramp
        elif name == "trapezoidal":
            return self._trapezoidal
        elif name == "s_curve":
            return self._s_curve
        elif name == "off":
            return lambda t: 0.0
        else:
            raise ValueError(f"Unknown motor profile: {c.profile!r}")

    def _constant(self, t: float) -> float:
        return self.config.max_torque

    def _ramp(self, t: float) -> float:
        """Linear ramp from 0 to max_torque over ramp_time, then hold."""
        c = self.config
        if t < c.ramp_time:
            return c.max_torque * t / c.ramp_time
        return c.max_torque

    def _trapezoidal(self, t: float) -> float:
        """Ramp up, hold, ramp down to zero.

        Timeline:
            [0, ramp_time)                      ramp up
            [ramp_time, ramp_time + hold_time)  hold at max
            [ramp_time + hold_time, 2*ramp_time + hold_time)  ramp down
            after that: zero
        """
        c = self.config
        t1 = c.ramp_time
        t2 = t1 + c.hold_time
        t3 = t2 + c.ramp_time

        if t < 0:
            return 0.0
        elif t < t1:
            return c.max_torque * t / t1
        elif t < t2:
            return c.max_torque
        elif t < t3:
            return c.max_torque * (1.0 - (t - t2) / c.ramp_time)
        else:
            return 0.0

    def _s_curve(self, t: float) -> float:
        """Smooth (sinusoidal) ramp up over ramp_time, then hold.

        τ(t) = τ_max * 0.5 * (1 - cos(π t / t_ramp))  for t < t_ramp
        """
        c = self.config
        if t < 0:
            return 0.0
        elif t < c.ramp_time:
            return c.max_torque * 0.5 * (1.0 - np.cos(np.pi * t / c.ramp_time))
        else:
            return c.max_torque
