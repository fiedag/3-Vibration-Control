"""3-axis accelerometer sensor model.

Each accelerometer at body-frame position r_s measures the specific
force (acceleration minus gravity, but in microgravity there's no
gravity so it's just the acceleration):

    a_meas = ω × (ω × r_s) + dω/dt × r_s + noise

The first term (centripetal) is the dominant signal — it's the
artificial gravity from spin.  The second term (Euler acceleration)
contains the nutation/precession dynamics that the control system
needs to observe.

In Level 1 (rigid body) there are no vibration components.
Level 2+ will add modal contributions.
"""

from __future__ import annotations

import numpy as np


class Accelerometer:
    """Single 3-axis accelerometer at a fixed body-frame position."""

    def __init__(self, position: np.ndarray, noise_std: float = 0.0):
        """
        Args:
            position: (3,) body-frame position [x, y, z] in metres.
            noise_std: standard deviation of white Gaussian noise
                       per axis (m/s²).
        """
        self.position = np.asarray(position, dtype=np.float64)
        self.noise_std = noise_std

    def measure(
        self,
        omega: np.ndarray,
        d_omega: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Compute accelerometer reading.

        Args:
            omega:   (3,) body-frame angular velocity (rad/s).
            d_omega: (3,) body-frame angular acceleration (rad/s²).
            rng:     random generator for noise.  None = no noise.

        Returns:
            (3,) measured acceleration in body frame (m/s²).
        """
        r = self.position

        # Centripetal: ω × (ω × r)
        centripetal = np.cross(omega, np.cross(omega, r))

        # Euler acceleration: dω/dt × r
        euler = np.cross(d_omega, r)

        a = centripetal + euler

        if self.noise_std > 0 and rng is not None:
            a = a + rng.normal(0.0, self.noise_std, size=3)

        return a


class AccelerometerArray:
    """Array of N accelerometers at different body-frame positions.

    Produces a flat observation vector of length 3*N.
    """

    def __init__(
        self,
        positions: np.ndarray,
        noise_std: float = 0.0,
    ):
        """
        Args:
            positions: (N, 3) body-frame positions.
            noise_std: noise std per axis, shared by all sensors.
        """
        self.positions = np.asarray(positions, dtype=np.float64)
        self.n_sensors = len(self.positions)
        self.noise_std = noise_std
        self._accels = [
            Accelerometer(pos, noise_std) for pos in self.positions
        ]

    def measure_all(
        self,
        omega: np.ndarray,
        d_omega: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Measure all accelerometers and return flat vector.

        Returns:
            (3*N,) concatenated readings: [a1_x, a1_y, a1_z, a2_x, ...].
        """
        readings = np.empty(3 * self.n_sensors)
        for i, accel in enumerate(self._accels):
            readings[3*i:3*i+3] = accel.measure(omega, d_omega, rng)
        return readings

    def measure_all_vectorised(
        self,
        omega: np.ndarray,
        d_omega: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Vectorised version — faster for many sensors.

        Returns:
            (3*N,) concatenated readings.
        """
        r = self.positions                            # (N, 3)

        # ω × r  for each sensor
        omega_cross_r = np.cross(omega, r)            # (N, 3)

        # ω × (ω × r)
        centripetal = np.cross(omega, omega_cross_r)  # (N, 3)

        # dω/dt × r
        euler = np.cross(d_omega, r)                  # (N, 3)

        a = centripetal + euler                       # (N, 3)

        if self.noise_std > 0 and rng is not None:
            a = a + rng.normal(0.0, self.noise_std, size=a.shape)

        return a.ravel()
