"""CLI analysis script: plot training telemetry from the database.

Usage:
    python -m habitat_sim.scripts.analyse_experiment --db habitat.db --experiment-id 1
    python -m habitat_sim.scripts.analyse_experiment --db habitat.db --list
"""

from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse and plot habitat-sim training telemetry."
    )
    parser.add_argument("--db", type=str, default="habitat.db",
                        help="Path to SQLite database.")
    parser.add_argument("--experiment-id", type=int, default=None,
                        help="Experiment ID to analyse.")
    parser.add_argument("--list", action="store_true",
                        help="List all experiments and exit.")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Directory for output plots (default: next to DB).")
    args = parser.parse_args()

    from habitat_sim.database.queries import (
        get_conservation_summary, get_nutation_curve,
        get_reward_curve, list_experiments,
    )

    if args.list:
        exps = list_experiments(args.db)
        if not exps:
            print("No experiments found.")
            return
        print(f"{'ID':>4}  {'Name':<30}  {'Episodes':>8}  {'Algorithm':<8}  Created")
        for e in exps:
            print(f"{e['id']:>4}  {e['name']:<30}  {e['n_episodes']:>8}  "
                  f"{e['algorithm']:<8}  {e['created_at'][:19]}")
        return

    if args.experiment_id is None:
        parser.error("--experiment-id is required (or use --list).")

    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Install with: pip install habitat-sim[viz]")
        return

    exp_id = args.experiment_id
    out_dir = args.out_dir or os.path.join(os.path.dirname(args.db) or ".", "plots")
    os.makedirs(out_dir, exist_ok=True)

    reward_data = get_reward_curve(args.db, exp_id)
    nutation_data = get_nutation_curve(args.db, exp_id)
    conservation_data = get_conservation_summary(args.db, exp_id)

    n_episodes = len(reward_data["episode_num"])
    if n_episodes == 0:
        print(f"No episodes found for experiment {exp_id}.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # Reward curve
    axes[0].plot(reward_data["episode_num"], reward_data["total_reward"],
                 linewidth=1.5, color="steelblue")
    axes[0].set_ylabel("Total reward")
    axes[0].set_title(f"Experiment {exp_id} — Training Telemetry ({n_episodes} episodes)")
    axes[0].grid(True, alpha=0.3)

    # Nutation
    axes[1].plot(nutation_data["episode_num"],
                 [v or 0.0 for v in nutation_data["final_nutation_deg"]],
                 linewidth=1.5, color="tomato")
    axes[1].set_ylabel("Final nutation (deg)")
    axes[1].grid(True, alpha=0.3)

    # Conservation violations
    axes[2].bar(conservation_data["episode_num"],
                conservation_data["h_violation_count"],
                color="orange", width=0.8)
    axes[2].set_ylabel("H violations")
    axes[2].set_xlabel("Episode")
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"experiment_{exp_id}_summary.png")
    plt.savefig(out_path, dpi=120)
    print(f"Plot saved to {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
