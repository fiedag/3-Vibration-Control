"""Prescribed crew and cargo movement schedules.

Mass transfer events move lumped masses between sectors over a specified
duration, providing smooth interpolation of the inertia tensor to avoid
discontinuities in the integrator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass
class MassTransfer:
    """A single mass transfer event between two sectors.

    Attributes:
        time:        start time (s) of the transfer.
        mass:        kg to move.
        from_sector: flat index of the source sector (0–35).
        to_sector:   flat index of the destination sector (0–35).
        duration:    time (s) over which the transfer occurs.
                     During this window the mass is linearly interpolated
                     from the source to the destination sector.
    """
    time: float
    mass: float
    from_sector: int
    to_sector: int
    duration: float = 30.0

    def __post_init__(self):
        if self.duration <= 0:
            raise ValueError("Transfer duration must be positive.")
        if self.from_sector == self.to_sector:
            raise ValueError("Source and destination sectors must differ.")


@dataclass
class StaticMass:
    """A fixed mass that sits in a sector for the entire simulation.

    Useful for initial crew/cargo placement.
    """
    sector: int
    mass: float


@dataclass
class MassScheduleConfig:
    """Configuration for a prescribed mass movement schedule."""
    static_masses: list[StaticMass] = field(default_factory=list)
    transfers: list[MassTransfer] = field(default_factory=list)


class MassSchedule:
    """Evaluates a prescribed mass distribution at any time t.

    Usage:
        schedule = MassSchedule(config, n_sectors=36)
        sector_masses = schedule.get_sector_masses(t)
    """

    def __init__(self, config: MassScheduleConfig, n_sectors: int = 36):
        self.n_sectors = n_sectors
        self.config = config

        # Sort transfers by start time for efficient evaluation
        self.transfers = sorted(config.transfers, key=lambda e: e.time)

        # Build base distribution from static masses
        self._base = np.zeros(n_sectors)
        for sm in config.static_masses:
            if not 0 <= sm.sector < n_sectors:
                raise ValueError(
                    f"Static mass sector {sm.sector} out of range [0, {n_sectors})"
                )
            self._base[sm.sector] += sm.mass

    def get_sector_masses(self, t: float) -> np.ndarray:
        """Return (n_sectors,) array of crew/cargo masses at time t.

        Mass transfers are smoothly interpolated: during a transfer,
        the mass linearly decreases at the source sector and linearly
        increases at the destination sector.
        """
        masses = self._base.copy()

        for xfer in self.transfers:
            t_start = xfer.time
            t_end = t_start + xfer.duration

            if t < t_start:
                # Transfer hasn't started yet — mass still at source
                masses[xfer.from_sector] += xfer.mass
            elif t >= t_end:
                # Transfer complete — mass at destination
                masses[xfer.to_sector] += xfer.mass
            else:
                # In progress — interpolate
                frac = (t - t_start) / xfer.duration
                masses[xfer.from_sector] += xfer.mass * (1.0 - frac)
                masses[xfer.to_sector] += xfer.mass * frac

        return masses

    def total_mass(self) -> float:
        """Total crew/cargo mass (should be constant regardless of time)."""
        total = sum(sm.mass for sm in self.config.static_masses)
        total += sum(xfer.mass for xfer in self.transfers)
        return total


# ---------------------------------------------------------------------------
# Convenience builders
# ---------------------------------------------------------------------------

def uniform_crew(mass_per_person: float, n_crew: int,
                 n_sectors: int = 36) -> MassScheduleConfig:
    """Place crew uniformly across all sectors."""
    mass_per_sector = mass_per_person * n_crew / n_sectors
    statics = [StaticMass(sector=i, mass=mass_per_sector)
               for i in range(n_sectors)]
    return MassScheduleConfig(static_masses=statics)


def single_imbalance(mass: float, sector: int = 0) -> MassScheduleConfig:
    """Place a single mass in one sector (worst-case static imbalance)."""
    return MassScheduleConfig(
        static_masses=[StaticMass(sector=sector, mass=mass)]
    )


def shift_change(
    mass_per_person: float,
    n_crew: int,
    from_sectors: Sequence[int],
    to_sectors: Sequence[int],
    start_time: float = 100.0,
    duration: float = 60.0,
) -> MassScheduleConfig:
    """Simulate a crew shift change: N people move between sector groups.

    Distributes crew evenly among from_sectors initially, then transfers
    them to to_sectors at start_time.

    Note: the MassTransfer objects handle mass placement at the source
    before the transfer starts, so no separate StaticMass entries are needed.
    """
    mass_each = mass_per_person * n_crew / len(from_sectors)
    transfers = []
    for src, dst in zip(from_sectors, to_sectors):
        transfers.append(MassTransfer(
            time=start_time, mass=mass_each,
            from_sector=src, to_sector=dst, duration=duration,
        ))
    return MassScheduleConfig(transfers=transfers)
