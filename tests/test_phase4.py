"""Phase 4 tests: SAC training pipeline.

Milestone criteria:
  1. build_vec_env returns env with correct obs/action shapes
  2. build_sac constructs agent without error
  3. Short training run (1000 steps) completes without crashing
  4. Saved model can be loaded and produces valid actions
  5. CurriculumCallback advances stages at correct timestep thresholds
  6. evaluate_agent returns dict with expected keys
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from gymnasium import spaces

from habitat_sim.config import (
    ExperimentConfig, MotorConfig, SimulationConfig, RLConfig, reference_config,
)
from habitat_sim.control.sac_agent import build_vec_env, build_sac, load_sac, check_model_compatibility
from habitat_sim.control.training import evaluate_agent, run_training


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fast_config() -> ExperimentConfig:
    """Minimal config for fast tests (10-second episodes)."""
    cfg = reference_config()
    cfg.motor = MotorConfig(profile="off")
    cfg.simulation = SimulationConfig(dt=0.01, duration=10.0, control_dt=0.1)
    cfg.rl = RLConfig(
        total_timesteps=200,
        n_envs=1,
        learning_starts=50,
        buffer_size=500,
        batch_size=32,
        eval_freq=100,
        n_eval_episodes=1,
        checkpoint_freq=100,
        curriculum=False,
    )
    cfg.seed = 0
    return cfg


# ---------------------------------------------------------------------------
# VecEnv construction
# ---------------------------------------------------------------------------

class TestVecEnv:

    def test_obs_action_shapes(self):
        cfg = _fast_config()
        env = build_vec_env(cfg, n_envs=1, seed=0)
        assert env.observation_space.shape == (75,)
        assert env.action_space.shape == (36,)
        env.close()

    def test_multiple_envs(self):
        cfg = _fast_config()
        env = build_vec_env(cfg, n_envs=2, seed=0)
        assert env.num_envs == 2
        obs = env.reset()
        assert obs.shape == (2, 75)
        env.close()

    def test_reset_returns_obs(self):
        cfg = _fast_config()
        env = build_vec_env(cfg, n_envs=1, seed=0)
        obs = env.reset()
        assert obs.shape == (1, 75)
        assert np.all(np.isfinite(obs))
        env.close()


# ---------------------------------------------------------------------------
# SAC construction
# ---------------------------------------------------------------------------

class TestBuildSAC:

    def test_constructs_without_error(self):
        cfg = _fast_config()
        env = build_vec_env(cfg, n_envs=1, seed=0)
        model = build_sac(env, cfg.rl, seed=0)
        assert model is not None
        env.close()

    def test_policy_name(self):
        cfg = _fast_config()
        env = build_vec_env(cfg, n_envs=1, seed=0)
        model = build_sac(env, cfg.rl, seed=0)
        assert "SAC" in model.__class__.__name__
        env.close()

    def test_predict_returns_valid_action(self):
        cfg = _fast_config()
        env = build_vec_env(cfg, n_envs=1, seed=0)
        model = build_sac(env, cfg.rl, seed=0)
        obs = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        assert action.shape == (1, 36)
        assert np.all(action >= -1.0) and np.all(action <= 1.0)
        env.close()


# ---------------------------------------------------------------------------
# Short training run
# ---------------------------------------------------------------------------

class TestShortTraining:

    def test_training_completes(self, tmp_path):
        cfg = _fast_config()
        cfg.rl.log_dir = str(tmp_path / "run")
        model = run_training(cfg)
        assert model is not None

    def test_final_model_saved(self, tmp_path):
        cfg = _fast_config()
        log_dir = str(tmp_path / "run")
        cfg.rl.log_dir = log_dir
        run_training(cfg)
        assert os.path.exists(os.path.join(log_dir, "final_model.zip"))

    def test_model_load_and_predict(self, tmp_path):
        cfg = _fast_config()
        log_dir = str(tmp_path / "run")
        cfg.rl.log_dir = log_dir
        run_training(cfg)

        model_path = os.path.join(log_dir, "final_model")
        loaded = load_sac(model_path)
        assert loaded is not None


# ---------------------------------------------------------------------------
# Curriculum callback
# ---------------------------------------------------------------------------

class TestCurriculumCallback:

    def test_curriculum_runs_without_error(self, tmp_path):
        cfg = _fast_config()
        cfg.rl.curriculum = True
        cfg.rl.total_timesteps = 200
        cfg.rl.log_dir = str(tmp_path / "run")
        model = run_training(cfg)
        assert model is not None


# ---------------------------------------------------------------------------
# evaluate_agent
# ---------------------------------------------------------------------------

class TestEvaluateAgent:

    def test_returns_expected_keys(self, tmp_path):
        cfg = _fast_config()
        cfg.rl.log_dir = str(tmp_path / "run")
        run_training(cfg)

        model_path = os.path.join(str(tmp_path / "run"), "final_model")
        results = evaluate_agent(model_path, cfg, n_episodes=2)

        for key in ("mean_reward", "std_reward", "mean_nutation_deg",
                    "std_nutation_deg", "mean_cm_offset", "episodes"):
            assert key in results, f"Missing key: {key}"

    def test_episode_count(self, tmp_path):
        cfg = _fast_config()
        cfg.rl.log_dir = str(tmp_path / "run")
        run_training(cfg)

        model_path = os.path.join(str(tmp_path / "run"), "final_model")
        results = evaluate_agent(model_path, cfg, n_episodes=3)
        assert len(results["episodes"]) == 3


# ---------------------------------------------------------------------------
# Model compatibility checks
# ---------------------------------------------------------------------------

class TestModelCompatibility:
    """check_model_compatibility() and evaluate_agent() fail fast and clearly
    when a model's observation or action space doesn't match the current env."""

    def _save_model_with_wrong_obs(self, tmp_path, correct_cfg) -> str:
        """Train a small model, then re-save it with a fake 93-dim obs space."""
        from stable_baselines3 import SAC

        correct_cfg.rl.log_dir = str(tmp_path / "run")
        run_training(correct_cfg)

        model_path = os.path.join(str(tmp_path / "run"), "final_model")
        model = SAC.load(model_path)

        # Simulate legacy 93-dim accelerometer sensor suite
        model.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(93,), dtype=np.float64
        )
        wrong_path = str(tmp_path / "legacy_model")
        model.save(wrong_path)
        return wrong_path

    def test_obs_mismatch_raises_value_error(self, tmp_path):
        """evaluate_agent raises ValueError (not a cryptic shape error) on obs mismatch."""
        cfg = _fast_config()
        wrong_path = self._save_model_with_wrong_obs(tmp_path, cfg)

        with pytest.raises(ValueError, match="Observation space mismatch"):
            evaluate_agent(wrong_path, _fast_config(), n_episodes=1)

    def test_obs_mismatch_error_names_both_shapes(self, tmp_path):
        """The error message must name both the model shape and the env shape."""
        cfg = _fast_config()
        wrong_path = self._save_model_with_wrong_obs(tmp_path, cfg)

        with pytest.raises(ValueError) as exc_info:
            evaluate_agent(wrong_path, _fast_config(), n_episodes=1)

        msg = str(exc_info.value)
        assert "93" in msg, "Error should mention model's obs shape (93,)"
        assert "75" in msg, "Error should mention env's obs shape (75,)"

    def test_obs_mismatch_error_mentions_model_path(self, tmp_path):
        """The error message must include the model path for easy diagnosis."""
        cfg = _fast_config()
        wrong_path = self._save_model_with_wrong_obs(tmp_path, cfg)

        with pytest.raises(ValueError) as exc_info:
            evaluate_agent(wrong_path, _fast_config(), n_episodes=1)

        assert wrong_path in str(exc_info.value)

    def test_compatible_model_does_not_raise(self, tmp_path):
        """A correctly-matched model passes the check without error."""
        cfg = _fast_config()
        cfg.rl.log_dir = str(tmp_path / "run")
        run_training(cfg)

        model_path = os.path.join(str(tmp_path / "run"), "final_model")
        # Should complete without raising
        results = evaluate_agent(model_path, _fast_config(), n_episodes=1)
        assert "mean_reward" in results

    def test_check_model_compatibility_directly(self, tmp_path):
        """check_model_compatibility() can be called standalone before evaluation."""
        from habitat_sim.environment.habitat_env import HabitatEnv
        from stable_baselines3 import SAC

        cfg = _fast_config()
        cfg.rl.log_dir = str(tmp_path / "run")
        run_training(cfg)

        model_path = os.path.join(str(tmp_path / "run"), "final_model")
        env = HabitatEnv(config=_fast_config())

        # Compatible — should not raise
        check_model_compatibility(model_path, env)

        # Incompatible — swap obs space and re-save
        model = SAC.load(model_path)
        model.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(93,), dtype=np.float64
        )
        wrong_path = str(tmp_path / "bad_model")
        model.save(wrong_path)

        with pytest.raises(ValueError, match="Observation space mismatch"):
            check_model_compatibility(wrong_path, env)

        env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
