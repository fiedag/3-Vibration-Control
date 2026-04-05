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


def check_model_compatibility(
    model_path: str,
    env: "gymnasium.Env | VecEnv",
) -> None:
    """Verify a saved model's observation and action spaces match the given env.

    Call this before running evaluate_agent() or any other evaluation loop to
    catch sensor-suite mismatches early, before they surface as cryptic array
    shape errors mid-episode.

    Reads space metadata *directly from the zip archive* (pure ``zipfile`` +
    ``json``, no network reconstruction) so the check never triggers a PyTorch
    ``RuntimeError`` due to weight-shape mismatches — even when the stored
    observation-space shape and the network weights are internally inconsistent
    (e.g. a model whose obs-space was patched for testing purposes).

    Args:
        model_path: Path to a .zip file produced by model.save() (with or
            without the .zip extension).
        env:        A live HabitatEnv (or VecEnv) built from the current config.

    Raises:
        ValueError: If observation or action space shapes do not match, with a
            message that names the expected vs actual shapes and hints at the
            likely cause (e.g. legacy accelerometer sensor suite).
        FileNotFoundError: If the zip file does not exist.
    """
    import json
    import zipfile

    # Accept paths with or without the .zip suffix (SB3 convention)
    zip_path = model_path if model_path.endswith(".zip") else model_path + ".zip"

    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Model file not found: {zip_path}")

    # SB3 stores metadata as a JSON blob named "data" inside the zip.
    # The observation_space and action_space entries each contain a "_shape"
    # key with the raw list of dimensions — no network reconstruction needed.
    with zipfile.ZipFile(zip_path, "r") as zf:
        data = json.loads(zf.read("data"))

    model_obs: tuple = tuple(data["observation_space"]["_shape"])
    model_act: tuple = tuple(data["action_space"]["_shape"])
    env_obs: tuple = env.observation_space.shape
    env_act: tuple = env.action_space.shape

    if model_obs != env_obs:
        raise ValueError(
            f"Observation space mismatch: model expects {model_obs} but "
            f"environment produces {env_obs}.\n"
            f"Model path: {model_path}\n"
            f"This usually means the model was trained with a different sensor "
            f"suite. Models trained before commit 85a3e3f used accelerometers "
            f"(93,); current code uses strain gauges (75,) and these are "
            f"incompatible. Retrain the model with the current sensor suite, or "
            f"load a model from runs/poisson_run_2/ or later."
        )

    if model_act != env_act:
        raise ValueError(
            f"Action space mismatch: model expects {model_act} but "
            f"environment has {env_act}.\n"
            f"Model path: {model_path}\n"
            f"Check that n_tanks_per_station and n_stations match between the "
            f"config used to train the model and the current config."
        )
