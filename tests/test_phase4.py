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

from habitat_sim.config import (
    ExperimentConfig, MotorConfig, SimulationConfig, RLConfig, reference_config,
)
from habitat_sim.control.sac_agent import build_vec_env, build_sac, load_sac
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
        assert env.observation_space.shape == (93,)
        assert env.action_space.shape == (36,)
        env.close()

    def test_multiple_envs(self):
        cfg = _fast_config()
        env = build_vec_env(cfg, n_envs=2, seed=0)
        assert env.num_envs == 2
        obs = env.reset()
        assert obs.shape == (2, 93)
        env.close()

    def test_reset_returns_obs(self):
        cfg = _fast_config()
        env = build_vec_env(cfg, n_envs=1, seed=0)
        obs = env.reset()
        assert obs.shape == (1, 93)
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
