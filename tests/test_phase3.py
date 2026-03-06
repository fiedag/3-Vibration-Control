"""Phase 3 tests: sensors and Gymnasium environment.

Milestone criteria:
  1. Accelerometer produces correct centripetal acceleration
  2. Accelerometer noise has correct statistics
  3. Mass tracker holds values between updates
  4. Mass tracker noise has correct statistics
  5. Sensor suite builds observation of correct dimension
  6. Gymnasium env reset returns correct shapes
  7. Gymnasium env step returns correct shapes
  8. Random agent runs full episode without crashing
  9. Observation vector contains physically plausible values
  10. Gymnasium env_checker passes
"""

from __future__ import annotations

import numpy as np
import pytest

from habitat_sim.config import (
    ExperimentConfig, MotorConfig, SimulationConfig,
    SensorConfig, reference_config,
)
from habitat_sim.sensors.accelerometer import Accelerometer, AccelerometerArray
from habitat_sim.sensors.mass_tracker import MassTracker
from habitat_sim.sensors.sensor_suite import SensorSuite
from habitat_sim.simulation.engine import SimulationEngine


# ===================================================================
# Accelerometer unit tests
# ===================================================================

class TestAccelerometer:

    def test_centripetal_on_x_axis(self):
        """Sensor at (R, 0, 0) spinning about z: a = -ω²R x̂."""
        R = 10.0
        omega_z = 0.2094
        accel = Accelerometer(position=np.array([R, 0.0, 0.0]))

        omega = np.array([0.0, 0.0, omega_z])
        d_omega = np.array([0.0, 0.0, 0.0])

        a = accel.measure(omega, d_omega, rng=None)

        # Centripetal: ω×(ω×r) = -ω²R x̂ for r on x-axis, ω on z-axis
        expected_ax = -omega_z**2 * R
        assert a[0] == pytest.approx(expected_ax, rel=1e-10)
        assert a[1] == pytest.approx(0.0, abs=1e-14)
        assert a[2] == pytest.approx(0.0, abs=1e-14)

    def test_centripetal_magnitude(self):
        """Centripetal acceleration magnitude should be ω²R."""
        R = 10.0
        omega_z = 0.2094
        theta = np.pi / 3  # 60°
        pos = np.array([R * np.cos(theta), R * np.sin(theta), 5.0])

        accel = Accelerometer(position=pos)
        omega = np.array([0.0, 0.0, omega_z])
        d_omega = np.zeros(3)

        a = accel.measure(omega, d_omega, rng=None)

        # Centripetal is in the xy-plane, magnitude ω²R
        a_xy = np.sqrt(a[0]**2 + a[1]**2)
        assert a_xy == pytest.approx(omega_z**2 * R, rel=1e-10)
        # z-component should be zero (no centripetal along spin axis)
        assert a[2] == pytest.approx(0.0, abs=1e-14)

    def test_euler_acceleration(self):
        """With angular acceleration, Euler term should appear."""
        R = 10.0
        pos = np.array([R, 0.0, 0.0])
        accel = Accelerometer(position=pos)

        omega = np.array([0.0, 0.0, 0.0])      # no spin
        d_omega = np.array([0.0, 0.0, 1.0])     # angular accel about z

        a = accel.measure(omega, d_omega, rng=None)

        # dω/dt × r = [0,0,1] × [R,0,0] = [0, R, 0]
        assert a[0] == pytest.approx(0.0, abs=1e-14)
        assert a[1] == pytest.approx(R, rel=1e-10)
        assert a[2] == pytest.approx(0.0, abs=1e-14)

    def test_noise_statistics(self):
        """Noise should have approximately correct mean and std."""
        pos = np.array([10.0, 0.0, 0.0])
        noise_std = 0.1
        accel = Accelerometer(position=pos, noise_std=noise_std)

        omega = np.zeros(3)
        d_omega = np.zeros(3)
        rng = np.random.default_rng(42)

        readings = np.array([
            accel.measure(omega, d_omega, rng) for _ in range(10000)
        ])

        # Mean should be near 0, std near noise_std
        for axis in range(3):
            assert abs(readings[:, axis].mean()) < 0.01
            assert readings[:, axis].std() == pytest.approx(noise_std, rel=0.1)


class TestAccelerometerArray:

    def test_vectorised_matches_sequential(self):
        """Vectorised and sequential should give identical results (no noise)."""
        positions = np.array([
            [10.0, 0.0, -5.0],
            [-10.0, 0.0, -5.0],
            [10.0, 0.0, 0.0],
            [-10.0, 0.0, 0.0],
            [10.0, 0.0, 5.0],
            [-10.0, 0.0, 5.0],
        ])
        arr = AccelerometerArray(positions, noise_std=0.0)

        omega = np.array([0.01, 0.005, 0.2094])
        d_omega = np.array([0.001, -0.002, 0.0])

        seq = arr.measure_all(omega, d_omega, rng=None)
        vec = arr.measure_all_vectorised(omega, d_omega, rng=None)

        np.testing.assert_allclose(seq, vec, atol=1e-14)

    def test_output_shape(self):
        positions = np.random.randn(6, 3)
        arr = AccelerometerArray(positions)
        omega = np.array([0.0, 0.0, 0.2])
        d_omega = np.zeros(3)
        result = arr.measure_all_vectorised(omega, d_omega)
        assert result.shape == (18,)


# ===================================================================
# Mass tracker tests
# ===================================================================

class TestMassTracker:

    def test_holds_between_updates(self):
        """Between update ticks, estimates should not change."""
        tracker = MassTracker(n_sectors=36, noise_std=0.0, update_rate=1.0)
        true_masses = np.ones(36) * 10.0

        m0 = tracker.estimate(0.0, true_masses)
        np.testing.assert_allclose(m0, true_masses)

        # Change true masses, but it's only 0.5s later (rate=1Hz)
        new_masses = np.ones(36) * 20.0
        m1 = tracker.estimate(0.5, new_masses)
        np.testing.assert_allclose(m1, true_masses)  # should still be old

        # At t=1.0, should update
        m2 = tracker.estimate(1.0, new_masses)
        np.testing.assert_allclose(m2, new_masses)

    def test_noise_added(self):
        """With noise, estimates should differ from true values."""
        tracker = MassTracker(n_sectors=36, noise_std=5.0, update_rate=100.0)
        rng = np.random.default_rng(42)
        true_masses = np.ones(36) * 50.0

        m = tracker.estimate(0.0, true_masses, rng)
        # Should not be exactly true_masses
        assert not np.allclose(m, true_masses)
        # But should be close (within a few sigma)
        assert np.abs(m - true_masses).max() < 30.0

    def test_non_negative(self):
        """Mass estimates should never be negative."""
        tracker = MassTracker(n_sectors=36, noise_std=10.0, update_rate=100.0)
        rng = np.random.default_rng(42)
        true_masses = np.ones(36) * 5.0  # close to zero, noise might go negative

        for i in range(100):
            m = tracker.estimate(float(i) * 0.01, true_masses, rng)
            assert (m >= 0).all(), f"Negative mass at step {i}: min = {m.min()}"

    def test_reset(self):
        """After reset, first call should produce fresh estimate."""
        tracker = MassTracker(n_sectors=36, noise_std=0.0, update_rate=1.0)
        true_masses = np.ones(36) * 10.0
        tracker.estimate(0.0, true_masses)

        tracker.reset()

        new_masses = np.ones(36) * 99.0
        m = tracker.estimate(0.0, new_masses)
        np.testing.assert_allclose(m, new_masses)


# ===================================================================
# Sensor suite tests
# ===================================================================

class TestSensorSuite:

    @staticmethod
    def _make_suite() -> SensorSuite:
        positions = np.array([
            [10.0, 0.0, -5.0], [-10.0, 0.0, -5.0],
            [10.0, 0.0, 0.0],  [-10.0, 0.0, 0.0],
            [10.0, 0.0, 5.0],  [-10.0, 0.0, 5.0],
        ])
        config = SensorConfig(
            n_accelerometers=6,
            accelerometer_noise_std=0.0,
            mass_tracker_noise_std=0.0,
            mass_tracker_update_rate=10.0,
        )
        return SensorSuite(config, positions, n_sectors=36, seed=42)

    def test_observation_dimension(self):
        suite = self._make_suite()
        assert suite.observation_dimension == 93  # 18 + 36 + 36 + 3

    def test_observation_shape(self):
        suite = self._make_suite()
        obs = suite.observe(
            t=0.0,
            omega=np.array([0.0, 0.0, 0.2]),
            d_omega=np.zeros(3),
            sector_masses=np.zeros(36),
            tank_masses=np.ones(36) * 50.0,
            manifold_masses=np.ones(3) * 10.0,
        )
        assert obs.shape == (93,)

    def test_observation_contains_tank_levels(self):
        """Tank fill levels should appear verbatim in observation."""
        suite = self._make_suite()
        tank_masses = np.arange(36, dtype=np.float64)
        manifold_masses = np.array([100.0, 200.0, 300.0])

        obs = suite.observe(
            t=0.0, omega=np.zeros(3), d_omega=np.zeros(3),
            sector_masses=np.zeros(36),
            tank_masses=tank_masses,
            manifold_masses=manifold_masses,
        )

        # Tanks at indices [54:90], manifolds at [90:93]
        np.testing.assert_array_equal(obs[54:90], tank_masses)
        np.testing.assert_array_equal(obs[90:93], manifold_masses)

    def test_spinning_produces_centripetal(self):
        """Accelerometer readings should show centripetal when spinning."""
        suite = self._make_suite()
        omega_z = 0.2
        obs = suite.observe(
            t=0.0,
            omega=np.array([0.0, 0.0, omega_z]),
            d_omega=np.zeros(3),
            sector_masses=np.zeros(36),
            tank_masses=np.ones(36) * 50.0,
            manifold_masses=np.ones(3) * 10.0,
        )

        # First accelerometer at (10, 0, -5): centripetal = -ω²×10 in x
        accel_readings = obs[:18].reshape(6, 3)
        expected_ax = -omega_z**2 * 10.0
        assert accel_readings[0, 0] == pytest.approx(expected_ax, rel=1e-8)


# ===================================================================
# Engine observation tests
# ===================================================================

class TestEngineObservation:

    def test_step_returns_observation(self):
        """engine.step should return (obs, info) tuple."""
        cfg = reference_config()
        cfg.motor = MotorConfig(profile="off")
        cfg.simulation = SimulationConfig(dt=0.01, duration=10.0, control_dt=0.1)

        engine = SimulationEngine(cfg)
        engine.state.omega[:] = [0.0, 0.0, 0.2094]

        action = np.zeros(36)
        obs, info = engine.step(action)

        assert obs.shape == (93,)
        assert isinstance(info, dict)

    def test_initial_observation(self):
        """get_initial_observation should return correct shape."""
        cfg = reference_config()
        engine = SimulationEngine(cfg)
        obs = engine.get_initial_observation()
        assert obs.shape == (93,)

    def test_reset_returns_observation(self):
        """engine.reset should return initial observation."""
        cfg = reference_config()
        engine = SimulationEngine(cfg)

        # Step a few times
        action = np.zeros(36)
        for _ in range(10):
            engine.step(action)

        # Reset
        obs = engine.reset(seed=123)
        assert obs.shape == (93,)
        assert engine.t == 0.0
        assert engine.state.omega[2] == pytest.approx(0.0)


# ===================================================================
# Gymnasium environment tests
# ===================================================================

class TestHabitatEnv:

    @staticmethod
    def _make_env(**kwargs):
        from habitat_sim.environment.habitat_env import HabitatEnv
        cfg = reference_config()
        cfg.motor = MotorConfig(profile="off")
        cfg.simulation = SimulationConfig(
            dt=0.01, duration=10.0, control_dt=0.1,
        )
        return HabitatEnv(config=cfg, **kwargs)

    def test_reset_shape(self):
        env = self._make_env()
        obs, info = env.reset(seed=42)
        assert obs.shape == (93,)
        assert isinstance(info, dict)

    def test_step_shape(self):
        env = self._make_env()
        env.reset(seed=42)
        action = np.zeros(36)
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape == (93,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_action_clipping(self):
        """Actions outside [-1, +1] should be clipped, not error."""
        env = self._make_env()
        env.reset(seed=42)
        action = np.ones(36) * 5.0  # way out of range
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (93,)  # should not crash

    def test_episode_terminates(self):
        """Episode should terminate after duration / control_dt steps."""
        env = self._make_env()
        env.reset(seed=42)

        n_max = int(10.0 / 0.1)  # 100 steps
        terminated = False
        for i in range(n_max + 10):
            action = np.zeros(36)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                break

        assert terminated
        assert info["step_count"] == n_max

    def test_random_agent_episode(self):
        """A random agent should complete an episode without crashing."""
        env = self._make_env()
        obs, _ = env.reset(seed=42)

        rng = np.random.default_rng(42)
        total_reward = 0.0
        steps = 0

        while True:
            action = rng.uniform(-1, 1, size=36)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            if terminated or truncated:
                break

        assert steps == 100  # 10s / 0.1s
        assert np.isfinite(total_reward)
        assert obs.shape == (93,)

    def test_reward_is_negative(self):
        """Reward should generally be negative (all terms are penalties)."""
        env = self._make_env()
        env.reset(seed=42)

        rng = np.random.default_rng(42)
        rewards = []
        for _ in range(20):
            action = rng.uniform(-1, 1, size=36)
            _, reward, _, _, _ = env.step(action)
            rewards.append(reward)

        # Most rewards should be negative (penalty-based)
        assert sum(r < 0 for r in rewards) >= 15

    def test_spaces_match(self):
        """obs/action shapes should match the declared spaces."""
        env = self._make_env()
        obs, _ = env.reset(seed=42)

        assert env.observation_space.contains(obs)

        action = env.action_space.sample()
        obs2, _, _, _, _ = env.step(action)
        assert env.observation_space.contains(obs2)


# ===================================================================
# Gymnasium env_checker
# ===================================================================

class TestGymnasiumChecker:

    def test_env_checker(self):
        """The official Gymnasium env_checker should pass."""
        from gymnasium.utils.env_checker import check_env
        from habitat_sim.environment.habitat_env import HabitatEnv

        cfg = reference_config()
        cfg.motor = MotorConfig(profile="off")
        cfg.simulation = SimulationConfig(
            dt=0.01, duration=5.0, control_dt=0.1,
        )

        env = HabitatEnv(config=cfg)

        # check_env raises if something is wrong
        # It prints warnings for non-critical issues
        check_env(env.unwrapped if hasattr(env, 'unwrapped') else env,
                   skip_render_check=True)


# ===================================================================
# Entry point
# ===================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
