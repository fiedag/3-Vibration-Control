"""CLI entry point for running/evaluating simulations: habitat-run."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run habitat simulation demos or evaluate a trained agent."
    )
    parser.add_argument("--demo", choices=["all", "torque-free", "imbalance",
                                            "tank", "random-agent"],
                        default="all", help="Which demo to run.")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained SAC model .zip for evaluation.")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to JSON ExperimentConfig file.")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of evaluation episodes (with --model).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from habitat_sim.config import ExperimentConfig, reference_config

    if args.config:
        with open(args.config) as f:
            cfg = ExperimentConfig.from_json(f.read())
    else:
        cfg = reference_config()
    cfg.seed = args.seed

    if args.model:
        # Evaluation mode
        from habitat_sim.control.training import evaluate_agent
        print(f"Evaluating model: {args.model}")
        results = evaluate_agent(args.model, cfg, n_episodes=args.episodes)
        print(f"Mean reward:      {results['mean_reward']:.4f} +/- {results['std_reward']:.4f}")
        print(f"Mean nutation:    {results['mean_nutation_deg']:.4f} deg")
        print(f"Mean CM offset:   {results['mean_cm_offset']:.4f} m")
    else:
        # Demo mode — reuse quick_sim logic
        import scripts.quick_sim as qs
        demo_map = {
            "torque-free": qs.demo_torque_free,
            "imbalance":   qs.demo_imbalance,
            "tank":        qs.demo_tank_correction,
            "random-agent": qs.demo_gymnasium_env,
        }
        if args.demo == "all":
            for fn in demo_map.values():
                fn()
        else:
            demo_map[args.demo]()


if __name__ == "__main__":
    main()
