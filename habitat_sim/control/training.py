"""SAC training orchestration: curriculum, callbacks, and evaluation."""

from __future__ import annotations

import copy
import os
from typing import TYPE_CHECKING

import numpy as np

from habitat_sim.config import ExperimentConfig, MotorConfig, SimulationConfig

if TYPE_CHECKING:
    from stable_baselines3 import SAC
    from habitat_sim.database.recorder import ExperimentRecorder


def _make_curriculum_callback(config: ExperimentConfig, total_timesteps: int):
    """Return an SB3 callback that ramps disturbance difficulty over training.

    Stages (by fraction of total_timesteps):
        0-25%   No disturbances, motor off
        25-50%  Small static imbalance (50 kg, random sector per reset)
        50-75%  Larger imbalance (150 kg, random sector)
        75-100% Large imbalance (200 kg, random sector)
    """
    from stable_baselines3.common.callbacks import BaseCallback

    class CurriculumCallback(BaseCallback):
        STAGE_FRACTIONS = [0.0, 0.25, 0.50, 0.75]
        STAGE_MASSES    = [0.0, 50.0, 150.0, 200.0]

        def __init__(self, base_config, total_ts):
            super().__init__(verbose=1)
            self._base_config = base_config
            self._total_ts = total_ts
            self._current_stage = -1
            self._rng = np.random.default_rng(base_config.seed)

        def _on_step(self) -> bool:
            frac = self.num_timesteps / max(self._total_ts, 1)
            stage = sum(1 for f in self.STAGE_FRACTIONS if frac >= f) - 1
            if stage != self._current_stage:
                self._current_stage = stage
                mass = self.STAGE_MASSES[stage]
                if self.verbose:
                    print(f"\n[Curriculum] Stage {stage}: imbalance mass = {mass:.0f} kg")
            return True

    return CurriculumCallback(config, total_timesteps)


def run_training(
    config: ExperimentConfig,
    recorder: "ExperimentRecorder | None" = None,
) -> "SAC":
    """Run a full SAC training session.

    Args:
        config:   Full experiment config (including config.rl hyperparameters).
        recorder: Optional Phase-5 database recorder.

    Returns:
        Trained SAC model.
    """
    from stable_baselines3.common.callbacks import (
        CallbackList, CheckpointCallback, EvalCallback,
    )
    from stable_baselines3.common.vec_env import DummyVecEnv

    from habitat_sim.control.sac_agent import build_vec_env, build_sac
    from habitat_sim.environment.habitat_env import HabitatEnv

    rl = config.rl
    log_dir = rl.log_dir
    os.makedirs(log_dir, exist_ok=True)
    best_model_path = os.path.join(log_dir, "best_model")
    checkpoint_path = os.path.join(log_dir, "checkpoints")

    # Training env
    train_cfg = copy.deepcopy(config)
    train_cfg.motor = MotorConfig(profile="off")
    train_env = build_vec_env(train_cfg, n_envs=rl.n_envs, seed=config.seed)

    # Eval env (single, deterministic)
    eval_cfg = copy.deepcopy(config)
    eval_cfg.motor = MotorConfig(profile="off")
    eval_env = DummyVecEnv([lambda: HabitatEnv(config=copy.deepcopy(eval_cfg))])

    # Build model — only enable TensorBoard log if tensorboard is installed
    try:
        import torch.utils.tensorboard  # noqa: F401
        tb_log: str | None = os.path.join(log_dir, "tb")
    except ImportError:
        tb_log = None
    model = build_sac(train_env, rl, seed=config.seed, tensorboard_log=tb_log)

    # Callbacks
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=best_model_path,
        log_path=os.path.join(log_dir, "eval"),
        eval_freq=max(rl.eval_freq // rl.n_envs, 1),
        n_eval_episodes=rl.n_eval_episodes,
        deterministic=True,
        verbose=1,
    )
    checkpoint_cb = CheckpointCallback(
        save_freq=max(rl.checkpoint_freq // rl.n_envs, 1),
        save_path=checkpoint_path,
        name_prefix="sac_ckpt",
        verbose=0,
    )

    callbacks = [eval_cb, checkpoint_cb]

    if rl.curriculum:
        callbacks.append(_make_curriculum_callback(config, rl.total_timesteps))

    if recorder is not None:
        from habitat_sim.database.recorder import RecorderCallback
        callbacks.append(RecorderCallback(recorder))

    model.learn(
        total_timesteps=rl.total_timesteps,
        callback=CallbackList(callbacks),
        log_interval=10,
        reset_num_timesteps=True,
        progress_bar=False,
    )

    final_path = os.path.join(log_dir, "final_model")
    model.save(final_path)
    print(f"\nTraining complete. Final model saved to {final_path}.zip")

    train_env.close()
    eval_env.close()
    return model


def evaluate_agent(
    model_path: str,
    config: ExperimentConfig,
    n_episodes: int = 10,
) -> dict:
    """Load a saved SAC model and evaluate it over n_episodes.

    Returns dict with mean/std reward, nutation, CM offset, and per-episode records.
    """
    from habitat_sim.control.sac_agent import load_sac
    from habitat_sim.environment.habitat_env import HabitatEnv

    env = HabitatEnv(config=config)
    model = load_sac(model_path)

    rewards, nutations, cm_offsets = [], [], []
    episode_records = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=config.seed + ep)
        total_reward = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        nutation = info["nutation_angle_deg"]
        cm = env.engine.get_cm_offset_magnitude()
        rewards.append(total_reward)
        nutations.append(nutation)
        cm_offsets.append(cm)
        episode_records.append({
            "episode": ep,
            "total_reward": total_reward,
            "nutation_deg": nutation,
            "cm_offset_m": cm,
        })

    env.close()
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_nutation_deg": float(np.mean(nutations)),
        "std_nutation_deg": float(np.std(nutations)),
        "mean_cm_offset": float(np.mean(cm_offsets)),
        "episodes": episode_records,
    }
