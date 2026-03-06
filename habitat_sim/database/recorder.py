"""Experiment recorder: context manager + SB3 callback for telemetry.

Usage with run_training():
    from habitat_sim.database.recorder import ExperimentRecorder
    from habitat_sim.control.training import run_training

    with ExperimentRecorder("habitat.db", "my_run", cfg) as recorder:
        run_training(cfg, recorder=recorder)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy.orm import Session

from habitat_sim.database.schema import Experiment, Episode, Timestep, get_engine

if TYPE_CHECKING:
    from habitat_sim.config import ExperimentConfig


class ExperimentRecorder:
    """Context manager that creates an Experiment row and records episodes.

    Args:
        db_path:         SQLite file path (or ":memory:" for tests).
        experiment_name: Human-readable label for this run.
        config:          ExperimentConfig to store as JSON.
    """

    def __init__(
        self,
        db_path: str,
        experiment_name: str,
        config: "ExperimentConfig",
    ):
        self._db_path = db_path
        self._name = experiment_name
        self._config = config
        self._engine = None
        self._experiment_id: int | None = None
        # Buffer: list of step-dicts accumulated during current episode
        self._step_buffer: list[dict] = []
        self._episode_num: int = 0

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "ExperimentRecorder":
        self._engine = get_engine(self._db_path)
        with Session(self._engine) as session:
            exp = Experiment(
                name=self._name,
                created_at=datetime.now(timezone.utc),
                config_json=self._config.to_json(),
                seed=self._config.seed,
                algorithm=self._config.rl.algorithm,
            )
            session.add(exp)
            session.commit()
            self._experiment_id = exp.id
        return self

    def __exit__(self, *_) -> None:
        # Flush any remaining buffered episode
        if self._step_buffer:
            self._flush_episode(reward=None)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_step(self, step_index: int, step_data: dict) -> None:
        """Buffer one control-step worth of telemetry.

        Expected keys in step_data:
            t, omega (3,), cm_offset_mag, total_water, kinetic_energy,
            reward (optional), n_violations
        """
        self._step_buffer.append({"step_index": step_index, **step_data})

    def record_episode(
        self,
        episode_num: int,
        steps_data: list[dict],
        engine_info: dict | None = None,
    ) -> None:
        """Write one complete episode to the database.

        Args:
            episode_num: 0-based episode counter.
            steps_data:  List of per-step dicts (keys as in record_step).
            engine_info: Optional dict from engine (nutation, cm_offset, omega_z).
        """
        if not steps_data:
            return

        total_reward = sum(s.get("reward") or 0.0 for s in steps_data)
        last = steps_data[-1]
        ei = engine_info or {}

        with Session(self._engine) as session:
            ep = Episode(
                experiment_id=self._experiment_id,
                episode_num=episode_num,
                n_steps=len(steps_data),
                total_reward=total_reward,
                final_nutation_deg=ei.get("final_nutation_deg"),
                final_cm_offset_mag=ei.get("final_cm_offset_mag"),
                final_omega_z=ei.get("final_omega_z"),
                h_violation_count=last.get("n_violations", 0),
            )
            session.add(ep)
            session.flush()  # get ep.id

            timesteps = [
                Timestep(
                    episode_id=ep.id,
                    step_index=s["step_index"],
                    t=s.get("t", 0.0),
                    omega_x=float(s.get("omega", [0, 0, 0])[0]),
                    omega_y=float(s.get("omega", [0, 0, 0])[1]),
                    omega_z=float(s.get("omega", [0, 0, 0])[2]),
                    cm_offset_mag=s.get("cm_offset_mag", 0.0),
                    total_water=s.get("total_water", 0.0),
                    kinetic_energy=s.get("kinetic_energy", 0.0),
                    reward=s.get("reward"),
                    n_violations=s.get("n_violations", 0),
                )
                for s in steps_data
            ]
            session.add_all(timesteps)
            session.commit()

    def _flush_episode(self, reward) -> None:
        """Flush self._step_buffer as a completed episode."""
        self.record_episode(self._episode_num, self._step_buffer)
        self._step_buffer = []
        self._episode_num += 1


# ---------------------------------------------------------------------------
# SB3 callback
# ---------------------------------------------------------------------------

try:
    from stable_baselines3.common.callbacks import BaseCallback

    class RecorderCallback(BaseCallback):
        """SB3 callback that records each episode to the telemetry database.

        Hooks into the training loop via _on_step() to buffer per-step data
        and _on_rollout_end() / episode-done detection to flush to DB.
        """

        def __init__(self, recorder: ExperimentRecorder):
            super().__init__(verbose=0)
            self._recorder = recorder
            # Per-environment step buffers keyed by env index
            self._env_bufs: dict[int, list[dict]] = {}
            self._ep_num: int = 0

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [{}])
            rewards = self.locals.get("rewards", [None])
            dones = self.locals.get("dones", [False])

            for i, info in enumerate(infos):
                buf = self._env_bufs.setdefault(i, [])
                reward = float(rewards[i]) if rewards[i] is not None else None
                buf.append({
                    "step_index": len(buf),
                    "t": info.get("t", self.num_timesteps * 0.1),
                    "omega": info.get("omega", [0.0, 0.0, 0.0]),
                    "cm_offset_mag": info.get("cm_offset_mag", 0.0),
                    "total_water": info.get("total_water", 0.0),
                    "kinetic_energy": info.get("kinetic_energy", 0.0),
                    "reward": reward,
                    "n_violations": info.get("n_violations", 0),
                })
                if dones[i]:
                    engine_info = {
                        "final_nutation_deg": info.get("nutation_angle_deg"),
                        "final_cm_offset_mag": info.get("cm_offset_mag"),
                        "final_omega_z": None,
                    }
                    self._recorder.record_episode(
                        self._ep_num, buf, engine_info,
                    )
                    self._env_bufs[i] = []
                    self._ep_num += 1
            return True

        def _on_training_end(self) -> None:
            for buf in self._env_bufs.values():
                if buf:
                    self._recorder.record_episode(self._ep_num, buf)
                    self._ep_num += 1
            self._env_bufs = {}

except ImportError:
    # stable-baselines3 not installed; RecorderCallback unavailable
    class RecorderCallback:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("stable-baselines3 is required for RecorderCallback.")
