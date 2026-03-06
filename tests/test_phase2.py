"""Phase 2 tests: disturbances and tank control.

Milestone criteria:
  1. MassSchedule produces correct sector masses over time
  2. Mass transfer interpolates smoothly
  3. Static imbalance produces correct CM offset
  4. Spinning imbalance produces conical whirl at spin frequency
  5. Tank redistribution can reduce CM offset
  6. Tank water conservation holds under active pumping
  7. Scenario integration works with engine
  8. Manifold axial equalisation converges
"""

from __future__ import annotations

import numpy as np
import pytest

from habitat_sim.config import (
    ExperimentConfig, HabitatConfig, MotorConfig, SimulationConfig,
    TankConfig, SectorConfig, reference_config,
)
from habitat_sim.disturbances.mass_schedule import (
    MassSchedule, MassScheduleConfig, MassTransfer, StaticMass,
    single_imbalance, uniform_crew, shift_change,
)
from habitat_sim.disturbances.scenario import Scenario, build_scenario
from habitat_sim.actuators.tank_system import TankSystem, compute_correction_target
from habitat_sim.simulation.engine import SimulationEngine
from habitat_sim.simulation.state import SimState


# ===================================================================
# Mass schedule tests
# ===================================================================

class TestMassSchedule:

    def test_static_mass(self):
        """A static mass should appear at its sector at all times."""
        cfg = MassScheduleConfig(
            static_masses=[StaticMass(sector=5, mass=80.0)]
        )
        sched = MassSchedule(cfg, n_sectors=36)
        m = sched.get_sector_masses(0.0)
        assert m[5] == pytest.approx(80.0)
        assert m.sum() == pytest.approx(80.0)

        m2 = sched.get_sector_masses(1000.0)
        np.testing.assert_array_equal(m, m2)

    def test_transfer_before_start(self):
        """Before transfer starts, mass is at source."""
        cfg = MassScheduleConfig(
            transfers=[MassTransfer(time=100.0, mass=50.0,
                                    from_sector=0, to_sector=6, duration=30.0)]
        )
        sched = MassSchedule(cfg, n_sectors=36)
        m = sched.get_sector_masses(50.0)
        assert m[0] == pytest.approx(50.0)
        assert m[6] == pytest.approx(0.0)

    def test_transfer_after_end(self):
        """After transfer completes, mass is at destination."""
        cfg = MassScheduleConfig(
            transfers=[MassTransfer(time=100.0, mass=50.0,
                                    from_sector=0, to_sector=6, duration=30.0)]
        )
        sched = MassSchedule(cfg, n_sectors=36)
        m = sched.get_sector_masses(200.0)
        assert m[0] == pytest.approx(0.0)
        assert m[6] == pytest.approx(50.0)

    def test_transfer_midpoint(self):
        """At midpoint, mass should be split 50/50."""
        cfg = MassScheduleConfig(
            transfers=[MassTransfer(time=100.0, mass=60.0,
                                    from_sector=0, to_sector=6, duration=20.0)]
        )
        sched = MassSchedule(cfg, n_sectors=36)
        m = sched.get_sector_masses(110.0)  # midpoint
        assert m[0] == pytest.approx(30.0)
        assert m[6] == pytest.approx(30.0)

    def test_total_mass_conserved_during_transfer(self):
        """Total mass should be constant throughout a transfer."""
        cfg = MassScheduleConfig(
            static_masses=[StaticMass(sector=3, mass=100.0)],
            transfers=[MassTransfer(time=50.0, mass=100.0,
                                    from_sector=3, to_sector=10, duration=40.0)]
        )
        sched = MassSchedule(cfg, n_sectors=36)
        for t in [0, 25, 50, 60, 70, 90, 100, 200]:
            m = sched.get_sector_masses(float(t))
            assert m.sum() == pytest.approx(200.0), f"Mass not conserved at t={t}"

    def test_multiple_transfers(self):
        """Multiple simultaneous transfers should all work."""
        cfg = MassScheduleConfig(
            transfers=[
                MassTransfer(time=0, mass=10, from_sector=0, to_sector=1, duration=10),
                MassTransfer(time=0, mass=20, from_sector=2, to_sector=3, duration=10),
            ]
        )
        sched = MassSchedule(cfg, n_sectors=36)
        m = sched.get_sector_masses(10.0)
        assert m[0] == pytest.approx(0.0)
        assert m[1] == pytest.approx(10.0)
        assert m[2] == pytest.approx(0.0)
        assert m[3] == pytest.approx(20.0)


# ===================================================================
# Convenience builder tests
# ===================================================================

class TestConvenienceBuilders:

    def test_uniform_crew(self):
        cfg = uniform_crew(mass_per_person=80.0, n_crew=20, n_sectors=36)
        sched = MassSchedule(cfg, n_sectors=36)
        m = sched.get_sector_masses(0.0)
        expected_per_sector = 80.0 * 20 / 36
        np.testing.assert_allclose(m, expected_per_sector, atol=1e-12)

    def test_single_imbalance(self):
        cfg = single_imbalance(mass=200.0, sector=5)
        sched = MassSchedule(cfg, n_sectors=36)
        m = sched.get_sector_masses(0.0)
        assert m[5] == pytest.approx(200.0)
        assert m.sum() == pytest.approx(200.0)

    def test_shift_change(self):
        cfg = shift_change(
            mass_per_person=80.0, n_crew=6,
            from_sectors=[0, 1, 2], to_sectors=[6, 7, 8],
            start_time=100.0, duration=60.0,
        )
        sched = MassSchedule(cfg, n_sectors=36)
        # Before shift
        m0 = sched.get_sector_masses(50.0)
        assert m0[0] > 0
        assert m0[6] == pytest.approx(0.0)
        # After shift
        m1 = sched.get_sector_masses(200.0)
        assert m1[0] == pytest.approx(0.0)
        assert m1[6] > 0


# ===================================================================
# Scenario tests
# ===================================================================

class TestScenario:

    def test_empty_scenario(self):
        scenario = Scenario(n_sectors=36)
        m = scenario.get_sector_masses(0.0)
        np.testing.assert_array_equal(m, np.zeros(36))

    def test_combined_sources(self):
        """Two mass sources should be summed."""
        cfg1 = MassScheduleConfig(static_masses=[StaticMass(0, 50)])
        cfg2 = MassScheduleConfig(static_masses=[StaticMass(0, 30)])
        s1 = MassSchedule(cfg1, 36)
        s2 = MassSchedule(cfg2, 36)
        scenario = Scenario([s1, s2], n_sectors=36)
        m = scenario.get_sector_masses(0.0)
        assert m[0] == pytest.approx(80.0)

    def test_build_scenario_from_dicts(self):
        """build_scenario should parse disturbance config dicts."""
        configs = [{
            "type": "mass_schedule",
            "static_masses": [{"sector": 3, "mass": 100.0}],
            "transfers": [],
        }]
        scenario = build_scenario(configs, n_sectors=36)
        m = scenario.get_sector_masses(0.0)
        assert m[3] == pytest.approx(100.0)


# ===================================================================
# Scenario integration with engine
# ===================================================================

class TestScenarioEngine:

    def test_engine_with_disturbance_config(self):
        """Engine should build and use a scenario from config."""
        cfg = reference_config()
        cfg.motor = MotorConfig(profile="off")
        cfg.simulation = SimulationConfig(dt=0.01, duration=10.0,
                                          control_dt=0.1)
        cfg.disturbances = [{
            "type": "mass_schedule",
            "static_masses": [{"sector": 0, "mass": 200.0}],
            "transfers": [],
        }]

        engine = SimulationEngine(cfg)
        engine.state.omega[:] = [0.0, 0.0, 0.2094]

        # Step using scenario (no override)
        action = np.zeros(36)
        for _ in range(100):
            engine.step(action)

        # CM should be offset (mass at sector 0 creates imbalance)
        cm_mag = engine.get_cm_offset_magnitude()
        assert cm_mag > 0.001, f"CM offset {cm_mag} too small — disturbance not applied"


# ===================================================================
# CM offset and conical whirl
# ===================================================================

class TestConicalWhirl:

    @staticmethod
    def _make_spinning_config(duration: float = 30.0) -> ExperimentConfig:
        cfg = reference_config()
        cfg.motor = MotorConfig(profile="off")
        cfg.simulation = SimulationConfig(
            dt=0.005, duration=duration, control_dt=0.05,
        )
        return cfg

    def test_imbalance_creates_cm_offset(self):
        """A mass in sector 0 should create a measurable CM offset."""
        cfg = self._make_spinning_config(duration=1.0)
        engine = SimulationEngine(cfg)

        sector_masses = np.zeros(36)
        sector_masses[0] = 200.0

        cm = engine.get_cm_offset()
        # Before any stepping, CM should already be offset
        # because CM is computed from geometry + current masses
        # Actually need to pass sector_masses to get_cm_offset...
        # Let me step once to populate _last_sector_masses
        engine.step_no_control(sector_masses)
        cm = engine.get_cm_offset()

        assert np.linalg.norm(cm[:2]) > 0.01, (
            f"CM offset {cm} too small for 200 kg imbalance at R=10m"
        )

    def test_spinning_imbalance_produces_wobble(self):
        """A spinning habitat with static imbalance should wobble."""
        cfg = self._make_spinning_config(duration=30.0)
        engine = SimulationEngine(cfg)
        engine.state.omega[:] = [0.0, 0.0, 0.2094]

        sector_masses = np.zeros(36)
        sector_masses[0] = 200.0

        # Record ω_x over time
        omega_x_history = []
        n_steps = int(cfg.simulation.duration / cfg.simulation.control_dt)
        for _ in range(n_steps):
            engine.step_no_control(sector_masses)
            omega_x_history.append(engine.state.omega[0])

        omega_x = np.array(omega_x_history)

        # Should see oscillation in ω_x (conical whirl)
        assert omega_x.std() > 1e-6, (
            "ω_x is constant — no conical whirl detected from mass imbalance"
        )

    def test_conical_whirl_frequency(self):
        """Wobble frequency should be near the spin frequency for
        a near-axisymmetric body (I_zz ≈ 2*I_xx → ω_nut ≈ ω_spin)."""
        cfg = self._make_spinning_config(duration=60.0)
        engine = SimulationEngine(cfg)
        omega_spin = 0.2094
        engine.state.omega[:] = [0.0, 0.0, omega_spin]

        sector_masses = np.zeros(36)
        sector_masses[0] = 200.0

        omega_x_history = []
        times = []
        dt_ctrl = cfg.simulation.control_dt
        n_steps = int(cfg.simulation.duration / dt_ctrl)
        for i in range(n_steps):
            engine.step_no_control(sector_masses)
            omega_x_history.append(engine.state.omega[0])
            times.append((i + 1) * dt_ctrl)

        omega_x = np.array(omega_x_history)

        # FFT to find dominant frequency
        from numpy.fft import rfft, rfftfreq
        N = len(omega_x)
        freqs = rfftfreq(N, d=dt_ctrl)
        spectrum = np.abs(rfft(omega_x))
        # Skip DC
        peak_idx = np.argmax(spectrum[1:]) + 1
        peak_freq = freqs[peak_idx]

        # Expected: nutation frequency ≈ spin_freq * (I_zz - I_xx)/I_xx
        spin_freq_hz = omega_spin / (2 * np.pi)
        # For this geometry, I_zz/I_xx ≈ some ratio; just check
        # peak is within factor of 3 of spin frequency
        assert 0.3 * spin_freq_hz < peak_freq < 3.0 * spin_freq_hz, (
            f"Peak freq {peak_freq:.4f} Hz not near spin freq "
            f"{spin_freq_hz:.4f} Hz"
        )


# ===================================================================
# Tank correction tests
# ===================================================================

class TestTankCorrection:

    def test_correction_target_computation(self):
        """compute_correction_target should produce valid targets."""
        cfg = reference_config()
        geom = SimulationEngine(cfg).geometry
        tank_pos = geom.compute_tank_positions(cfg.tanks)

        target = compute_correction_target(
            desired_cm_x=-0.1, desired_cm_y=0.0,
            tank_positions=tank_pos,
            total_water=cfg.tanks.total_water_mass,
            tank_capacity=cfg.tanks.tank_capacity,
        )

        # Should sum to total water
        assert target.sum() == pytest.approx(cfg.tanks.total_water_mass, rel=1e-4)
        # All non-negative
        assert (target >= -1e-10).all()
        # None exceed capacity
        assert (target <= cfg.tanks.tank_capacity + 1e-10).all()

    def test_tank_pumping_reduces_cm_offset(self):
        """Active tank control should reduce the CM offset from an imbalance.

        Strategy: place a mass at sector 0, then pump water to the
        opposite side to counteract.
        """
        cfg = reference_config()
        cfg.motor = MotorConfig(profile="off")
        cfg.simulation = SimulationConfig(
            dt=0.01, duration=30.0, control_dt=0.1,
        )

        engine = SimulationEngine(cfg)
        engine.state.omega[:] = [0.0, 0.0, 0.2094]

        # 200 kg at sector 0 (angular index 0, axial index 0)
        sector_masses = np.zeros(36)
        sector_masses[0] = 200.0

        # Measure initial CM offset
        engine.step_no_control(sector_masses)
        cm_initial = engine.get_cm_offset_magnitude()

        # Build a valve command: drain sector-0 tanks, fill sector-6 tanks
        # (sector 6 is at 180° from sector 0)
        # Tank layout mirrors sector layout, so tank index 6 at each
        # station is opposite tank 0.
        action = np.zeros(36)
        for station in range(3):
            base = station * 12
            action[base + 0] = -1.0   # drain tank at 0°
            action[base + 6] = +1.0   # fill tank at 180°

        # Run with active pumping
        n_steps = int(29.0 / cfg.simulation.control_dt)  # remaining time
        for _ in range(n_steps):
            engine.step(action, sector_masses_override=sector_masses)

        cm_final = engine.get_cm_offset_magnitude()

        # CM offset should have decreased
        assert cm_final < cm_initial, (
            f"Tank correction failed: CM went from {cm_initial:.4f} "
            f"to {cm_final:.4f} m"
        )

    def test_water_conserved_under_active_pumping(self):
        """Total water must be exactly conserved even with active pumping."""
        cfg = reference_config()
        cfg.motor = MotorConfig(profile="off")
        cfg.simulation = SimulationConfig(
            dt=0.01, duration=20.0, control_dt=0.1,
        )

        engine = SimulationEngine(cfg)
        engine.state.omega[:] = [0.0, 0.0, 0.2094]
        initial_water = engine.state.total_water()

        # Random valve commands
        rng = np.random.default_rng(42)
        sector_masses = np.zeros(36)
        n_steps = int(cfg.simulation.duration / cfg.simulation.control_dt)

        for _ in range(n_steps):
            action = rng.uniform(-1, 1, size=36)
            engine.step(action, sector_masses_override=sector_masses)

        final_water = engine.state.total_water()
        assert final_water == pytest.approx(initial_water, abs=1e-8), (
            f"Water drift: {final_water - initial_water:.2e} kg"
        )


# ===================================================================
# Tank system constraint tests
# ===================================================================

class TestTankSystem:

    def test_enforce_constraints_clips_tanks(self):
        cfg = reference_config()
        state = SimState(cfg)

        ts = TankSystem(cfg.tanks)

        # Set one tank over capacity, one negative
        state.tank_masses[0] = cfg.tanks.tank_capacity + 50.0
        state.tank_masses[1] = -10.0

        ts.enforce_constraints(state)

        assert state.tank_masses[0] <= cfg.tanks.tank_capacity
        assert state.tank_masses[1] >= 0.0

    def test_enforce_constraints_preserves_total(self):
        cfg = reference_config()
        state = SimState(cfg)
        ts = TankSystem(cfg.tanks)

        # Perturb water total
        state.manifold_masses[0] += 5.0
        ts.enforce_constraints(state)

        assert state.total_water() == pytest.approx(
            cfg.tanks.total_water_mass, abs=1e-10
        )


# ===================================================================
# Manifold axial equalisation
# ===================================================================

class TestAxialEqualisation:

    def test_manifolds_converge_to_equal(self):
        """With no valve activity, manifolds should equalise over time."""
        cfg = reference_config()
        cfg.motor = MotorConfig(profile="off")
        cfg.simulation = SimulationConfig(
            dt=0.01, duration=120.0, control_dt=0.1,
        )
        # Increase axial gain for faster test convergence
        cfg.tanks.k_axial = 0.5

        engine = SimulationEngine(cfg)
        engine.state.omega[:] = [0.0, 0.0, 0.2094]

        # Create imbalanced manifolds
        engine.state.manifold_masses[:] = [100.0, 30.0, 30.0]
        # Adjust tanks to maintain total water
        tank_total = cfg.tanks.total_water_mass - 160.0
        engine.state.tank_masses[:] = tank_total / 36.0

        sector_masses = np.zeros(36)
        n_steps = int(cfg.simulation.duration / cfg.simulation.control_dt)
        for _ in range(n_steps):
            engine.step_no_control(sector_masses)

        # Manifolds should be closer to equal
        manifold_std = engine.state.manifold_masses.std()
        assert manifold_std < 10.0, (
            f"Manifolds still imbalanced after 120 s: "
            f"std = {manifold_std:.2f} kg, "
            f"levels = {engine.state.manifold_masses}"
        )


# ===================================================================
# Entry point
# ===================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
