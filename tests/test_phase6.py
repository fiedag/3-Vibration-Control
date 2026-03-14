"""Phase 6 tests: stochastic disturbances and toroid geometry.

Milestone criteria:
  Stochastic disturbances:
    1. PoissonCrewDisturbance total mass conserved at all sampled times
    2. At least one sector-change event occurs in a 1000 s simulation
    3. Linear interpolation is smooth (no discontinuities between samples)
    4. MicroImpactDisturbance produces non-zero output at an impact time
    5. build_scenario_from_stochastic_config constructs correct sources

  Toroid geometry:
    6. structural_mass() matches analytical formula
    7. compute_structural_inertia() is diagonal with I_xx == I_yy
    8. I_zz > I_xx (spin-axis has more inertia for R >> r)
    9. Sector positions are all at radius R in z=0 plane
   10. Toroid config can be passed to SimulationEngine and simulates without error
"""

from __future__ import annotations

import numpy as np
import pytest

from habitat_sim.config import (
    reference_config,
    HabitatConfig,
    StochasticConfig,
)
from habitat_sim.disturbances.stochastic import (
    PoissonCrewDisturbance,
    MicroImpactDisturbance,
)
from habitat_sim.disturbances.scenario import build_scenario_from_stochastic_config
from habitat_sim.geometry.toroid import ToroidGeometry


# ---------------------------------------------------------------------------
# Stochastic disturbances
# ---------------------------------------------------------------------------

class TestPoissonCrew:

    def _make_crew(self, **kwargs) -> PoissonCrewDisturbance:
        defaults = dict(
            n_sectors=36,
            n_crew=6,
            mass_per_person=80.0,
            lambda_rate=0.1,   # high rate so events occur in short windows
            transfer_duration=5.0,
            seed=42,
        )
        defaults.update(kwargs)
        return PoissonCrewDisturbance(**defaults)

    def test_total_mass_conserved(self):
        """Total mass equals n_crew * mass_per_person at all sampled times."""
        crew = self._make_crew()
        expected = 6 * 80.0
        for t in [0.0, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0]:
            masses = crew.get_sector_masses(t)
            assert masses.sum() == pytest.approx(expected, rel=1e-6), (
                f"Mass not conserved at t={t}: got {masses.sum()}"
            )

    def test_sector_change_occurs(self):
        """At least one sector configuration change in 1000 s."""
        crew = self._make_crew(lambda_rate=0.1)
        m0 = crew.get_sector_masses(0.0).copy()
        changed = False
        for t in np.linspace(1.0, 1000.0, 200):
            m = crew.get_sector_masses(t)
            if not np.allclose(m, m0):
                changed = True
                break
        assert changed, "No sector changes detected in 1000 s"

    def test_smooth_interpolation(self):
        """No large discontinuity between closely-spaced samples."""
        crew = self._make_crew(lambda_rate=0.01, transfer_duration=30.0)
        times = np.linspace(0.0, 200.0, 2000)
        prev = crew.get_sector_masses(times[0])
        for t in times[1:]:
            curr = crew.get_sector_masses(t)
            max_jump = np.abs(curr - prev).max()
            # A smooth move of 80 kg over 30 s sampled at dt=0.1 s
            # gives max step = 80 * 0.1/30 = 0.267 kg; allow 2x margin.
            assert max_jump < 1.0, (
                f"Discontinuity of {max_jump:.3f} kg at t={t:.2f}"
            )
            prev = curr


class TestMicroImpact:

    def test_nonzero_at_impact_time(self):
        """Mass is non-zero during an impact window."""
        dist = MicroImpactDisturbance(
            n_sectors=36,
            rate=1.0,         # 1 impact/s -- guaranteed to have impacts
            mass_std=1.0,
            duration=2.0,
            seed=0,
        )
        # Sample densely; at least one window must be non-zero
        found = False
        for t in np.linspace(0.0, 20.0, 200):
            if dist.get_sector_masses(t).sum() > 0.0:
                found = True
                break
        assert found, "MicroImpactDisturbance produced no non-zero output"

    def test_zero_outside_impact(self):
        """Returns zero array before any impacts."""
        dist = MicroImpactDisturbance(
            n_sectors=36,
            rate=0.001,
            mass_std=0.1,
            duration=1.0,
            seed=99,
        )
        # t=0 is before the first impact (first inter-event time >= 1/0.001=1000 s expected)
        m = dist.get_sector_masses(0.0)
        assert m.sum() == pytest.approx(0.0)


class TestBuildScenarioFromStochastic:

    def test_poisson_crew_source_added(self):
        cfg = StochasticConfig(poisson_crew=True, n_crew=4, lambda_rate=0.05)
        scenario = build_scenario_from_stochastic_config(cfg, n_sectors=36, seed=0)
        assert len(scenario.sources) == 1
        assert isinstance(scenario.sources[0], PoissonCrewDisturbance)

    def test_micro_impact_source_added(self):
        cfg = StochasticConfig(micro_impacts=True)
        scenario = build_scenario_from_stochastic_config(cfg, n_sectors=36, seed=0)
        assert len(scenario.sources) == 1
        assert isinstance(scenario.sources[0], MicroImpactDisturbance)

    def test_both_sources_added(self):
        cfg = StochasticConfig(poisson_crew=True, micro_impacts=True)
        scenario = build_scenario_from_stochastic_config(cfg, n_sectors=36, seed=0)
        assert len(scenario.sources) == 2

    def test_empty_config_no_sources(self):
        cfg = StochasticConfig()  # both disabled by default
        scenario = build_scenario_from_stochastic_config(cfg, n_sectors=36)
        assert len(scenario.sources) == 0

    def test_crew_total_mass_correct(self):
        """Scenario mass equals n_crew * mass_per_person."""
        cfg = StochasticConfig(
            poisson_crew=True, n_crew=6, mass_per_person=80.0,
        )
        scenario = build_scenario_from_stochastic_config(cfg, n_sectors=36)
        assert scenario.get_sector_masses(0.0).sum() == pytest.approx(480.0)


# ---------------------------------------------------------------------------
# Toroid geometry
# ---------------------------------------------------------------------------

def _make_toroid(R: float = 50.0, r: float = 5.0) -> ToroidGeometry:
    c = reference_config().habitat
    # Override shape and radii
    from dataclasses import replace
    c = replace(c, shape="toroid", radius=R, minor_radius=r)
    return ToroidGeometry(c)


class TestToroidGeometry:

    def test_structural_mass_formula(self):
        """m = wall_density * 4*pi^2*R*r * wall_thickness."""
        tg = _make_toroid(R=50.0, r=5.0)
        c = tg.config
        expected = c.wall_density * 4.0 * np.pi**2 * c.radius * c.minor_radius * c.wall_thickness
        assert tg.structural_mass() == pytest.approx(expected, rel=1e-9)

    def test_inertia_is_diagonal(self):
        """Off-diagonal elements are zero."""
        tg = _make_toroid()
        I = tg.compute_structural_inertia()
        assert I.shape == (3, 3)
        off_diag = I - np.diag(np.diag(I))
        assert np.allclose(off_diag, 0.0)

    def test_ixx_equals_iyy(self):
        """I_xx == I_yy by axisymmetry."""
        tg = _make_toroid()
        I = tg.compute_structural_inertia()
        assert I[0, 0] == pytest.approx(I[1, 1])

    def test_izz_greater_than_ixx(self):
        """I_zz > I_xx when R >> r (spin axis dominates)."""
        tg = _make_toroid(R=50.0, r=2.0)
        I = tg.compute_structural_inertia()
        assert I[2, 2] > I[0, 0]

    def test_sector_positions_at_major_radius(self):
        """All sector positions lie at radius R in the z=0 plane."""
        tg = _make_toroid(R=50.0, r=5.0)
        c = reference_config()
        positions = tg.compute_sector_positions(c.sectors)
        radii = np.sqrt(positions[:, 0]**2 + positions[:, 1]**2)
        assert np.allclose(radii, tg.config.radius, rtol=1e-9)
        assert np.allclose(positions[:, 2], 0.0)

    def test_sector_count_equals_n_angular(self):
        """Toroid returns n_angular sectors (not n_angular * n_axial)."""
        tg = _make_toroid()
        c = reference_config()
        positions = tg.compute_sector_positions(c.sectors)
        assert positions.shape[0] == c.sectors.n_angular

    def test_inertia_analytical_values(self):
        """Check I_zz and I_xx against closed-form expressions."""
        R, r = 50.0, 5.0
        tg = _make_toroid(R=R, r=r)
        m = tg.structural_mass()
        I = tg.compute_structural_inertia()
        assert I[2, 2] == pytest.approx(m * (2*R**2 + 3*r**2) / 2.0, rel=1e-9)
        assert I[0, 0] == pytest.approx(m * (2*R**2 + 5*r**2) / 4.0, rel=1e-9)

    def test_toroid_simulates_without_error(self):
        """SimulationEngine with toroid shape runs a few steps."""
        from habitat_sim.simulation.engine import SimulationEngine
        from habitat_sim.config import SectorConfig, TankConfig
        from dataclasses import replace

        cfg = reference_config()
        # Toroid has no axial extent -- use n_axial=1 so n_total == n_angular.
        cfg.habitat = replace(cfg.habitat, shape="toroid", minor_radius=5.0)
        cfg.sectors = SectorConfig(n_angular=12, n_axial=1)
        cfg.tanks = replace(cfg.tanks, n_tanks_per_station=12, n_stations=1)

        engine = SimulationEngine(cfg)
        state = engine.reset(seed=0)
        assert state is not None

        # Run a few steps with zero control action
        action = np.zeros(cfg.tanks.n_tanks_total)
        for _ in range(3):
            obs, info = engine.step(action)
        assert np.all(np.isfinite(obs))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
