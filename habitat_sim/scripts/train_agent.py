"""CLI entry point for SAC training: habitat-train."""

from __future__ import annotations

import argparse
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a SAC agent on the rotating habitat simulation."
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to JSON ExperimentConfig file.")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Override rl.total_timesteps.")
    parser.add_argument("--n-envs", type=int, default=None,
                        help="Override rl.n_envs (parallel environments).")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="Override rl.log_dir.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed.")
    parser.add_argument("--episode-duration", type=float, default=None,
                        help="Episode duration in seconds (default: 60 s for training).")
    parser.add_argument("--no-curriculum", action="store_true",
                        help="Disable curriculum learning.")
    parser.add_argument("--db", type=str, default=None,
                        help="Path to SQLite database for telemetry recording.")
    parser.add_argument("--experiment-name", type=str, default="sac_run",
                        help="Name for this experiment in the database.")
    args = parser.parse_args()

    from habitat_sim.config import ExperimentConfig, reference_config

    if args.config:
        with open(args.config) as f:
            cfg = ExperimentConfig.from_json(f.read())
    else:
        cfg = reference_config()

    # Apply CLI overrides
    if args.timesteps is not None:
        cfg.rl.total_timesteps = args.timesteps
    if args.n_envs is not None:
        cfg.rl.n_envs = args.n_envs
    if args.log_dir is not None:
        cfg.rl.log_dir = args.log_dir
    if args.seed is not None:
        cfg.seed = args.seed
    if args.episode_duration is not None:
        cfg.simulation.duration = args.episode_duration
    if args.no_curriculum:
        cfg.rl.curriculum = False

    from habitat_sim.control.training import run_training

    if args.db:
        from habitat_sim.database.recorder import ExperimentRecorder
        with ExperimentRecorder(args.db, args.experiment_name, cfg) as recorder:
            run_training(cfg, recorder=recorder)
    else:
        run_training(cfg)


if __name__ == "__main__":
    main()
