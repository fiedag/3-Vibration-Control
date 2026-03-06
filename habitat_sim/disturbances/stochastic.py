"""Stochastic disturbance sources for the habitat simulation.

Two sources are provided:
  - PoissonCrewDisturbance: crew members randomly walking between sectors
  - MicroImpactDisturbance: brief transient mass perturbations
"""

from __future__ import annotations

import numpy as np


class PoissonCrewDisturbance:
    """Models crew members who randomly walk between sectors.

    Crew movements follow a Poisson process: inter-event times are drawn
    from an Exponential distribution. Each move is a smooth linear
    interpolation over transfer_duration seconds, preserving total mass.

    Parameters
    ----------
    n_sectors:
        Number of sectors in the habitat ring.
    n_crew:
        Number of crew members.
    mass_per_person:
        Mass of each crew member in kg (default 80 kg).
    lambda_rate:
        Mean sector transitions per second per crew member (default 0.01).
    transfer_duration:
        Duration of each smooth transition in seconds (default 30 s).
    seed:
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        n_sectors: int = 36,
        n_crew: int = 6,
        mass_per_person: float = 80.0,
        lambda_rate: float = 0.01,
        transfer_duration: float = 30.0,
        seed: int = 0,
    ) -> None:
        self.n_sectors = n_sectors
        self.n_crew = n_crew
        self.mass_per_person = mass_per_person
        self.lambda_rate = lambda_rate
        self.transfer_duration = transfer_duration
        self._rng = np.random.default_rng(seed)

        # Current sector for each crew member (integer index)
        self._positions: list[int] = [
            int(self._rng.integers(0, n_sectors)) for _ in range(n_crew)
        ]

        # Active transfers: list of (crew_idx, from_sector, to_sector, t_start, t_end)
        self._active_transfers: list[tuple[int, int, int, float, float]] = []

        # Pre-generated event schedule: list of (t_event, crew_idx, to_sector)
        self._event_schedule: list[tuple[float, int, int]] = []
        self._schedule_horizon: float = 0.0
        self._last_t: float = 0.0

        # Pre-generate initial schedule
        self._extend_schedule(horizon=1000.0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extend_schedule(self, horizon: float) -> None:
        """Generate Poisson events for all crew up to the given horizon."""
        t_start = self._schedule_horizon

        for crew_idx in range(self.n_crew):
            # Find the last event time for this crew member
            crew_events = [
                e for e in self._event_schedule if e[1] == crew_idx
            ]
            t = crew_events[-1][0] if crew_events else t_start

            while t < horizon:
                dt = self._rng.exponential(1.0 / self.lambda_rate)
                t += dt
                if t > horizon:
                    break
                # Choose a different sector
                current = self._positions[crew_idx]
                choices = [s for s in range(self.n_sectors) if s != current]
                dest = int(self._rng.choice(choices))
                self._event_schedule.append((t, crew_idx, dest))

        self._schedule_horizon = horizon

    def _process_events_up_to(self, t: float) -> None:
        """Process all scheduled events up to time t, creating transfers."""
        # Extend schedule if needed
        if t > self._schedule_horizon * 0.9:
            self._extend_schedule(horizon=self._schedule_horizon + 1000.0)

        # Sort events by time (only once after extending)
        self._event_schedule.sort(key=lambda e: e[0])

        # Consume events up to t
        remaining = []
        for event in self._event_schedule:
            t_ev, crew_idx, dest = event
            if t_ev <= t:
                from_sector = self._positions[crew_idx]
                if from_sector != dest:
                    t_end = t_ev + self.transfer_duration
                    self._active_transfers.append(
                        (crew_idx, from_sector, dest, t_ev, t_end)
                    )
                    self._positions[crew_idx] = dest
            else:
                remaining.append(event)
        self._event_schedule = remaining

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_sector_masses(self, t: float) -> np.ndarray:
        """Return (n_sectors,) mass array at time t.

        Total mass is always n_crew * mass_per_person.
        """
        # Process any new events
        if t > self._last_t:
            self._process_events_up_to(t)
            self._last_t = t

        masses = np.zeros(self.n_sectors)

        # Base masses from current positions (will be adjusted for transfers)
        for crew_idx, sector in enumerate(self._positions):
            masses[sector] += self.mass_per_person

        # Adjust for active (in-progress) transfers
        completed = []
        for transfer in self._active_transfers:
            crew_idx, from_sector, to_sector, t_start, t_end = transfer
            if t >= t_end:
                completed.append(transfer)
                continue
            if t < t_start:
                continue
            # Linear interpolation: fraction moved
            alpha = (t - t_start) / (t_end - t_start)
            move_mass = self.mass_per_person * alpha
            # Current position already has full mass; subtract partial from
            # to_sector and add to from_sector to undo, then re-apply smoothly.
            # Actually: crew member is currently counted at to_sector (new pos).
            # We adjust: to_sector gets alpha fraction, from_sector gets (1-alpha).
            masses[to_sector] -= self.mass_per_person  # remove from new pos
            masses[from_sector] += self.mass_per_person * (1.0 - alpha)
            masses[to_sector] += self.mass_per_person * alpha

        # Remove completed transfers
        for t_obj in completed:
            self._active_transfers.remove(t_obj)

        return masses


class MicroImpactDisturbance:
    """Models brief transient mass perturbations (micro-impacts).

    Each impact applies a small mass to a random sector for a fixed
    duration. Impacts follow a Poisson process.

    Parameters
    ----------
    n_sectors:
        Number of sectors.
    rate:
        Impacts per second (default 0.001).
    mass_std:
        Std of impact mass in kg (default 0.1). Actual mass ~ |Normal(0, mass_std)|.
    duration:
        Duration of each impact in seconds (default 1.0 s).
    seed:
        RNG seed.
    """

    def __init__(
        self,
        n_sectors: int = 36,
        rate: float = 0.001,
        mass_std: float = 0.1,
        duration: float = 1.0,
        seed: int = 0,
    ) -> None:
        self.n_sectors = n_sectors
        self.rate = rate
        self.mass_std = mass_std
        self.duration = duration
        self._rng = np.random.default_rng(seed)

        # Pre-generate impact schedule: list of (t_impact, sector, mass)
        self._impacts: list[tuple[float, int, float]] = []
        self._schedule_horizon: float = 0.0
        self._last_t: float = 0.0

        self._extend_schedule(horizon=1000.0)

    def _extend_schedule(self, horizon: float) -> None:
        t = self._schedule_horizon
        while t < horizon:
            dt = self._rng.exponential(1.0 / self.rate)
            t += dt
            if t > horizon:
                break
            sector = int(self._rng.integers(0, self.n_sectors))
            mass = float(abs(self._rng.normal(0.0, self.mass_std)))
            self._impacts.append((t, sector, mass))
        self._schedule_horizon = horizon

    def get_sector_masses(self, t: float) -> np.ndarray:
        """Return (n_sectors,) mass array at time t."""
        if t > self._schedule_horizon * 0.9:
            self._extend_schedule(horizon=self._schedule_horizon + 1000.0)

        if t > self._last_t:
            self._last_t = t

        masses = np.zeros(self.n_sectors)
        for t_impact, sector, mass in self._impacts:
            t_end = t_impact + self.duration
            if t_impact <= t < t_end:
                masses[sector] += mass
        return masses
