"""Sector mass tracking system.

Provides noisy estimates of crew/cargo mass in each sector.  The mass
tracker operates at a lower update rate than the accelerometers (e.g.
1 Hz vs 100 Hz), so observations hold their previous value between
updates.
"""

from __future__ import annotations

import numpy as np


class MassTracker:
    """Noisy, rate-limited sector mass estimator.

    Between updates, the tracker holds the last measured values.
    At each update tick, it samples the true masses with additive noise.
    """

    def __init__(
        self,
        n_sectors: int = 36,
        noise_std: float = 1.0,
        update_rate: float = 1.0,
    ):
        """
        Args:
            n_sectors:   number of sectors to track.
            noise_std:   standard deviation of mass estimate noise (kg).
            update_rate: how often the tracker refreshes (Hz).
        """
        self.n_sectors = n_sectors
        self.noise_std = noise_std
        self.update_interval = 1.0 / update_rate if update_rate > 0 else 0.0

        # Internal state
        self._last_estimate = np.zeros(n_sectors)
        self._last_update_time = -1e30   # force update on first call

    def estimate(
        self,
        t: float,
        true_masses: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Return current sector mass estimates.

        If enough time has elapsed since the last update, refreshes
        the estimates with new noisy measurements.  Otherwise returns
        the held values.

        Args:
            t:           current simulation time (s).
            true_masses: (n_sectors,) actual crew/cargo masses.
            rng:         random generator for noise.

        Returns:
            (n_sectors,) estimated masses (kg).
        """
        if t - self._last_update_time >= self.update_interval:
            self._last_estimate = true_masses.copy()
            if self.noise_std > 0 and rng is not None:
                noise = rng.normal(0.0, self.noise_std, size=self.n_sectors)
                self._last_estimate += noise
                # Clip to non-negative (mass can't be negative)
                np.clip(self._last_estimate, 0.0, None,
                        out=self._last_estimate)
            self._last_update_time = t

        return self._last_estimate

    def reset(self) -> None:
        """Reset internal state (call at episode start)."""
        self._last_estimate = np.zeros(self.n_sectors)
        self._last_update_time = -1e30
