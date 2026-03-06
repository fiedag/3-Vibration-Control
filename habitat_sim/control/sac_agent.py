"""SAC agent construction and vectorised environment helpers.

Wraps stable-baselines3 SAC with project-specific defaults and
provides factory functions for building parallel environments.
"""

from __future__ import annotations

import os
import sys
from typing import Callable

import numpy as np

try:
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False

from habitat_sim.config import ExperimentConfig, RLConfig, MotorConfig, SimulationConfig
from habitat_sim.environment.habitat_env import HabitatEnv


def _require_sb3() -> None:
    if not HAS_SB3:
        raise ImportError(
            "stable-baselines3 is required for RL training.\n"
            "Install with: pip install habitat-sim[rl]"
        )


def make_env(config: ExperimentConfig, rank: int, seed: int) -> Callable:
    """Return a factory function that creates a seeded HabitatEnv instance.

    Designed for use with DummyVecEnv / SubprocVecEnv.
    """
    def _init() -> HabitatEnv:
        import copy
        env_cfg = copy.deepcopy(config)
        env_cfg.seed = seed + rank
        env = HabitatEnv(config=env_cfg)
        env.reset(seed=seed + rank)
        return env
    return _init


def build_vec_env(
    config: ExperimentConfig,
    n_envs: int,
    seed: int = 42,
) -> "VecEnv":
    """Build a vectorised environment with n_envs parallel HabitatEnv instances.

    Uses SubprocVecEnv on non-Windows platforms for true parallelism;
    falls back to DummyVecEnv on Windows (subprocess spawn overhead is high).
    """
    _require_sb3()
    fns = [make_env(config, rank=i, seed=seed) for i in range(n_envs)]
    if sys.platform == "win32" or n_envs == 1:
        return DummyVecEnv(fns)
    return SubprocVecEnv(fns)


def build_sac(
    env: "VecEnv",
    rl_config: RLConfig,
    seed: int = 42,
    tensorboard_log: str | None = None,
) -> "SAC":
    """Construct an SAC agent for the given vectorised environment.

    Args:
        env:            Vectorised gymnasium environment.
        rl_config:      Hyperparameter config from ExperimentConfig.rl.
        seed:           Random seed.
        tensorboard_log: Directory for TensorBoard logs (None to disable).

    Returns:
        Configured but untrained SAC instance.
    """
    _require_sb3()
    import torch.nn as nn

    policy_kwargs = {
        "net_arch": rl_config.net_arch,
        "activation_fn": nn.ReLU,
    }

    return SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=rl_config.learning_rate,
        buffer_size=rl_config.buffer_size,
        batch_size=rl_config.batch_size,
        learning_starts=rl_config.learning_starts,
        gamma=rl_config.gamma,
        tau=rl_config.tau,
        ent_coef=rl_config.ent_coef,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=seed,
        tensorboard_log=tensorboard_log,
    )


def load_sac(model_path: str, env: "VecEnv | None" = None) -> "SAC":
    """Load a saved SAC model from disk.

    Args:
        model_path: Path to .zip file produced by model.save().
        env:        Optional env to attach (needed for further training).

    Returns:
        Loaded SAC model.
    """
    _require_sb3()
    return SAC.load(model_path, env=env)
