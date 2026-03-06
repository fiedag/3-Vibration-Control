"""Scenario: combines multiple disturbance sources.

A Scenario holds one or more disturbance sources (MassSchedule,
future stochastic models, etc.) and provides the combined sector
masses at any time t.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from habitat_sim.disturbances.mass_schedule import MassSchedule, MassScheduleConfig


class MassSource(Protocol):
    """Protocol for anything that can provide sector masses at time t."""

    def get_sector_masses(self, t: float) -> np.ndarray:
        ...


class Scenario:
    """Combines multiple mass disturbance sources.

    The sector masses from each source are summed element-wise.
    """

    def __init__(self, sources: list[MassSource] | None = None,
                 n_sectors: int = 36):
        self.n_sectors = n_sectors
        self.sources: list[MassSource] = sources or []

    def get_sector_masses(self, t: float) -> np.ndarray:
        """Return combined (n_sectors,) mass array at time t."""
        masses = np.zeros(self.n_sectors)
        for source in self.sources:
            masses += source.get_sector_masses(t)
        return masses

    def add_source(self, source: MassSource) -> None:
        self.sources.append(source)


# ---------------------------------------------------------------------------
# Factory: build Scenario from disturbance config list
# ---------------------------------------------------------------------------

def build_scenario(
    disturbance_configs: list[dict],
    n_sectors: int = 36,
) -> Scenario:
    """Build a Scenario from a list of disturbance config dicts.

    Each dict should have a 'type' key. Currently supported:
        - type: "mass_schedule"  → builds a MassSchedule

    This is the entry point used by SimulationEngine when constructed
    from an ExperimentConfig.
    """
    sources: list[MassSource] = []

    for dc in disturbance_configs:
        dtype = dc.get("type", "mass_schedule")

        if dtype == "mass_schedule":
            from habitat_sim.disturbances.mass_schedule import (
                StaticMass, MassTransfer,
            )
            statics = [StaticMass(**s) for s in dc.get("static_masses", [])]
            transfers = [MassTransfer(**t) for t in dc.get("transfers", [])]
            cfg = MassScheduleConfig(static_masses=statics,
                                     transfers=transfers)
            sources.append(MassSchedule(cfg, n_sectors=n_sectors))
        elif dtype == "poisson_crew":
            from habitat_sim.disturbances.stochastic import PoissonCrewDisturbance
            params = {k: v for k, v in dc.items() if k != "type"}
            sources.append(PoissonCrewDisturbance(n_sectors=n_sectors, **params))
        elif dtype == "micro_impact":
            from habitat_sim.disturbances.stochastic import MicroImpactDisturbance
            params = {k: v for k, v in dc.items() if k != "type"}
            sources.append(MicroImpactDisturbance(n_sectors=n_sectors, **params))
        else:
            raise ValueError(f"Unknown disturbance type: {dtype!r}")

    return Scenario(sources, n_sectors=n_sectors)


def build_scenario_from_stochastic_config(
    stochastic_cfg,
    n_sectors: int = 36,
    seed: int = 0,
) -> Scenario:
    """Build a Scenario from a StochasticConfig dataclass.

    Convenience wrapper used when constructing environments from an
    ExperimentConfig that has stochastic disturbances enabled.
    """
    from habitat_sim.disturbances.stochastic import (
        PoissonCrewDisturbance,
        MicroImpactDisturbance,
    )
    sources: list[MassSource] = []

    if stochastic_cfg.poisson_crew:
        sources.append(
            PoissonCrewDisturbance(
                n_sectors=n_sectors,
                n_crew=stochastic_cfg.n_crew,
                mass_per_person=stochastic_cfg.mass_per_person,
                lambda_rate=stochastic_cfg.lambda_rate,
                transfer_duration=stochastic_cfg.transfer_duration,
                seed=seed,
            )
        )

    if stochastic_cfg.micro_impacts:
        sources.append(
            MicroImpactDisturbance(
                n_sectors=n_sectors,
                rate=stochastic_cfg.impact_rate,
                mass_std=stochastic_cfg.impact_mass_std,
                duration=stochastic_cfg.impact_duration,
                seed=seed + 1,
            )
        )

    return Scenario(sources, n_sectors=n_sectors)
