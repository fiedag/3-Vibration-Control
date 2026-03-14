"""Phase 3 tests: sensors and Gymnasium environment.

Milestone criteria:
  1. Strain gauge reads correct centripetal force for steady spin
  2. Strain gauge force is proportional to sector mass
  3. Strain gauge noise has correct statistics
  4. Strain gauge detects wobble (asymmetric forces for non-zero ω_x)
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
from habitat_sim.sensors.strain_gauge import StrainGaugeArray
from habitat_sim.sensors.sensor_suite import SensorSuite
from habitat_sim.simulation.engine import SimulationEngine


# ===================================================================
# Strain gauge unit tests
# ===================================================================

class TestStrainGauge:

    @staticmethod
    def _ring_positions(n: int = 12, R: float = 10.0) -> np.ndarray:
        """12 sector centroids evenly spaced on a ring of radius R."""
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        positions = np.zeros((n, 3))
        positions[:, 0] = R * np.cos(angles)
        positions[:, 1] = R * np.sin(angles)
        return positions

    def test_centripetal_force_steady_spin(self):
        """For pure spin (ω_z) a sector of mass m should feel m·ω²·R."""
        R = 10.0
        omega_z = 0.2094
        m = 80.0  # kg

        positions = self._ring_positions(12, R)
        gauges = StrainGaugeArray(positions, noise_std=0.0)

        omega = np.array([0.0, 0.0, omega_z])
        d_omega = np.zeros(3)
        masses = np.full(12, m)

        forces = gauges.measure(omega, d_omega, masses, rng=None)

        expected = m * omega_z**2 * R
        np.testing.assert_allclose(forces, expected, rtol=1e-10)

    def test_force_proportional_to_mass(self):
        """Force should scale linearly with sector mass."""
        R = 10.0
        omega_z = 0.2094
        positions = self._ring_positions(12, R)
        gauges = StrainGaugeArray(positions, noise_std=0.0)

        omega = np.array([0.0, 0.0, omega_z])
        d_omega = np.zeros(3)

        masses_1 = np.ones(12) * 50.0
        masses_2 = np.ones(12) * 100.0

        f1 = gauges.measure(omega, d_omega, masses_1)
        f2 = gauges.measure(omega, d_omega, masses_2)

        np.testing.assert_allclose(f2, 2 * f1, rtol=1e-10)

    def test_zero_mass_zero_force(self):
        """Empty sectors should register zero force."""
        positions = self._ring_positions(12)
        gauges = StrainGaugeArray(positions, noise_std=0.0)

        omega = np.array([0.0, 0.0, 0.2094])
        d_omega = np.zeros(3)
        masses = np.zeros(12)

        forces = gauges.measure(omega, d_omega, masses, rng=None)
        np.testing.assert_allclose(forces, 0.0, atol=1e-15)

    def test_wobble_creates_asymmetry(self):
        """Non-zero transverse ω should break the rotational symmetry."""
        R = 10.0
        positions = self._ring_positions(12, R)
        gauges = StrainGaugeArray(positions, noise_std=0.0)

        masses = np.ones(12) * 80.0
        d_omega = np.zeros(3)

        # Pure spin — all forces equal
        f_steady = gauges.measure(np.array([0.0, 0.0, 0.2094]), d_omega, masses)
        assert np.allclose(f_steady, f_steady[0], rtol=1e-8)

        # Spin + wobble — forces vary around the ring
        f_wobble = gauges.measure(np.array([0.05, 0.0, 0.2094]), d_omega, masses)
        assert not np.allclose(f_wobble, f_wobble[0], rtol=1e-3)

    def test_noise_statistics(self):
        """Noise should have approximately correct mean and std."""
        R = 10.0
        noise_std = 5.0
        positions = self._ring_positions(1, R)
        gauges = StrainGaugeArray(positions, noise_std=noise_std)

        omega = np.array([0.0, 0.0, 0.2094])
        d_omega = np.zeros(3)
        masses = np.array([80.0])
        rng = np.random.default_rng(42)

        readings = np.array([
            gauges.measure(omega, d_omega, masses, rng)[0]
            for _ in range(10000)
        ])

        expected_mean = masses[0] * 0.2094**2 * R
        assert abs(readings.mean() - expected_mean) < 1.0
        assert readings.std() == pytest.approx(noise_std, rel=0.1)

    def test_output_shape(self):
        positions = self._ring_positions(36)
        gauges = StrainGaugeArray(positions)
        omega = np.array([0.0, 0.0, 0.2094])
        d_omega = np.zeros(3)
        masses = np.ones(36) * 50.0
        result = gauges.measure(omega, d_omega, masses)
        assert result.shape == (36,)


# ===================================================================
# Sensor suite tests
# ===================================================================

class TestSensorSuite:

    @staticmethod
    def _sector_positions(n_angular: int = 12, n_axial: int = 3,
                          R: float = 10.0, L: float = 20.0) -> np.ndarray:
        n = n_angular * n_axial
        positions = np.zeros((n, 3))
        angles = np.linspace(0, 2 * np.pi, n_angular, endpoint=False)
        axial_centers = np.linspace(-L / 2, L / 2, n_axial + 1)
        axial_centers = (axial_centers[:-1] + axial_centers[1:]) / 2
        idx = 0
        for j in range(n_axial):
            for i in range(n_angular):
                positions[idx] = [R * np.cos(angles[i]),
                                   R * np.sin(angles[i]),
                                   axial_centers[j]]
                idx += 1
        return positions

    @staticmethod
    def _make_suite() -> SensorSuite:
        positions = TestSensorSuite._sector_positions()
        config = SensorConfig(strain_gauge_noise_std=0.0)
        return SensorSuite(config, positions, n_sectors=36, seed=42)

    def test_observation_dimension(self):
        suite = self._make_suite()
        assert suite.observation_dimension == 75  # 36 + 36 + 3

    def test_observation_shape(self):
        suite = self._make_suite()
        obs = suite.observe(
            omega=np.array([0.0, 0.0, 0.2]),
            d_omega=np.zeros(3),
            sector_masses=np.ones(36) * 50.0,
            tank_masses=np.ones(36) * 50.0,
            manifold_masses=np.ones(3) * 10.0,
        )
        assert obs.shape == (75,)

    def test_observation_contains_tank_levels(self):
        """Tank fill levels should appear verbatim in observation."""
        suite = self._make_suite()
        tank_masses = np.arange(36, dtype=np.float64)
        manifold_masses = np.array([100.0, 200.0, 300.0])

        obs = suite.observe(
            omega=np.zeros(3), d_omega=np.zeros(3),
            sector_masses=np.zeros(36),
            tank_masses=tank_masses,
            manifold_masses=manifold_masses,
        )

        # Tanks at indices [36:72], manifolds at [72:75]
        np.testing.assert_array_equal(obs[36:72], tank_masses)
        np.testing.assert_array_equal(obs[72:75], manifold_masses)

    def test_spinning_produces_positive_forces(self):
        """All strain gauge readings should be positive during spin."""
        suite = self._make_suite()
        masses = np.ones(36) * 80.0
        obs = suite.observe(
            omega=np.array([0.0, 0.0, 0.2094]),
            d_omega=np.zeros(3),
            sector_masses=masses,
            tank_masses=np.ones(36) * 50.0,
            manifold_masses=np.ones(3) * 10.0,
        )
        # All strain gauge readings should be > 0
        assert (obs[:36] > 0).all()


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

        assert obs.shape == (75,)
        assert isinstance(info, dict)

    def test_initial_observation(self):
        """get_initial_observation should return correct shape."""
        cfg = reference_config()
        engine = SimulationEngine(cfg)
        obs = engine.get_initial_observation()
        assert obs.shape == (75,)

    def test_reset_returns_observation(self):
        """engine.reset should return initial observation."""
        cfg = reference_config()
        engine = SimulationEngine(cfg)

        action = np.zeros(36)
        for _ in range(10):
            engine.step(action)

        obs = engine.reset(seed=123)
        assert obs.shape == (75,)
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
        assert obs.shape == (75,)
        assert isinstance(info, dict)

    def test_step_shape(self):
        env = self._make_env()
        env.reset(seed=42)
        action = np.zeros(36)
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape == (75,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_action_clipping(self):
        """Actions outside [-1, +1] should be clipped, not error."""
        env = self._make_env()
        env.reset(seed=42)
        action = np.ones(36) * 5.0
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (75,)

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
        assert obs.shape == (75,)

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

        check_env(env.unwrapped if hasattr(env, 'unwrapped') else env,
                   skip_render_check=True)


# ===================================================================
# Entry point
# ===================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
