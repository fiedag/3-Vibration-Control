"""Numerical integrator for the state vector.

RK4 with post-step quaternion normalisation.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def rk4_step(
    deriv_fn: Callable[..., np.ndarray],
    t: float,
    x: np.ndarray,
    dt: float,
    *args,
) -> np.ndarray:
    """Classical 4th-order Runge-Kutta step.

    Args:
        deriv_fn: f(t, x, *args) -> dx/dt  as flat numpy array.
        t:  current time.
        x:  current state vector (flat).
        dt: time step.
        *args: extra arguments forwarded to deriv_fn.

    Returns:
        Updated state vector x(t + dt).  Caller must normalise the
        quaternion component afterwards.
    """
    k1 = deriv_fn(t, x, *args)
    k2 = deriv_fn(t + dt / 2, x + dt / 2 * k1, *args)
    k3 = deriv_fn(t + dt / 2, x + dt / 2 * k2, *args)
    k4 = deriv_fn(t + dt, x + dt * k3, *args)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
