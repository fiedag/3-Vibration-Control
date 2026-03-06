#!/usr/bin/env python3
"""Quick simulation demo — run from the project root.

Demonstrates:
  1. Torque-free spinning habitat (conservation check)
  2. Mass imbalance producing conical whirl
  3. Tank correction reducing the wobble
  4. Gymnasium environment with a random agent

Usage:
    python scripts/quick_sim.py
"""

import time
import numpy as np

from habitat_sim.config import (
    ExperimentConfig, MotorConfig, SimulationConfig, reference_config,
)
from habitat_sim.disturbances.mass_schedule import single_imbalance, MassSchedule
from habitat_sim.simulation.engine import SimulationEngine


def divider(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def demo_torque_free():
    """1. Torque-free spinning habitat — verify conservation."""
    divider("Demo 1: Torque-Free Spinner (60 s)")

    cfg = reference_config()
    cfg.motor = MotorConfig(profile="off")
    cfg.simulation = SimulationConfig(dt=0.01, duration=60.0, control_dt=0.1)

    engine = SimulationEngine(cfg)
    engine.state.omega[:] = [0.0, 0.0, 0.2094]  # ~2 rpm

    sector_masses = np.zeros(36)
    initial_water = engine.state.total_water()

    t0 = time.perf_counter()
    n_steps = int(60.0 / 0.1)
    for _ in range(n_steps):
        engine.step_no_control(sector_masses)
    elapsed = time.perf_counter() - t0

    info = engine.monitor.get_info()
    print(f"  Wall clock:       {elapsed:.2f} s  ({60.0/elapsed:.1f}x real-time)")
    print(f"  Final omega_z:    {engine.state.omega[2]:.6f} rad/s  (initial: 0.209400)")
    print(f"  |omega_x|:        {abs(engine.state.omega[0]):.2e}  (should be ~0)")
    print(f"  |omega_y|:        {abs(engine.state.omega[1]):.2e}  (should be ~0)")
    print(f"  |q| norm:         {np.linalg.norm(engine.state.quaternion):.15f}")
    print(f"  Water drift:      {engine.state.total_water() - initial_water:.2e} kg")
    print(f"  H violations:     {info.get('n_violations', 0)}")


def demo_imbalance():
    """2. Mass imbalance producing conical whirl."""
    divider("Demo 2: 200 kg Imbalance at Sector 0 (30 s)")

    cfg = reference_config()
    cfg.motor = MotorConfig(profile="off")
    cfg.simulation = SimulationConfig(dt=0.005, duration=30.0, control_dt=0.05)

    engine = SimulationEngine(cfg)
    engine.state.omega[:] = [0.0, 0.0, 0.2094]

    sector_masses = np.zeros(36)
    sector_masses[0] = 200.0

    nutation_history = []
    n_steps = int(30.0 / 0.05)
    for _ in range(n_steps):
        engine.step_no_control(sector_masses)
        nutation_history.append(engine.get_nutation_angle())

    nutation = np.array(nutation_history)
    cm = engine.get_cm_offset()

    print(f"  CM offset:        [{cm[0]:.4f}, {cm[1]:.4f}, {cm[2]:.4f}] m")
    print(f"  |CM offset|:      {np.linalg.norm(cm):.4f} m")
    print(f"  Nutation range:   {nutation.min():.4f} - {nutation.max():.4f} deg")
    print(f"  Nutation std:     {nutation.std():.4f} deg")
    print(f"  Conical whirl:    {'YES' if nutation.std() > 0.001 else 'NO'}")


def demo_tank_correction():
    """3. Tank correction reducing CM offset."""
    divider("Demo 3: Tank Correction (30 s)")

    cfg = reference_config()
    cfg.motor = MotorConfig(profile="off")
    cfg.simulation = SimulationConfig(dt=0.01, duration=30.0, control_dt=0.1)

    engine = SimulationEngine(cfg)
    engine.state.omega[:] = [0.0, 0.0, 0.2094]

    sector_masses = np.zeros(36)
    sector_masses[0] = 200.0

    # Measure initial CM offset
    engine.step_no_control(sector_masses)
    cm_initial = engine.get_cm_offset_magnitude()

    # Apply opposing tank correction: drain at 0°, fill at 180°
    action = np.zeros(36)
    for station in range(3):
        base = station * 12
        action[base + 0] = -1.0
        action[base + 6] = +1.0

    n_steps = int(29.0 / 0.1)
    for _ in range(n_steps):
        engine.step(action, sector_masses_override=sector_masses)

    cm_final = engine.get_cm_offset_magnitude()

    print(f"  CM offset before: {cm_initial:.4f} m")
    print(f"  CM offset after:  {cm_final:.4f} m")
    print(f"  Reduction:        {(1 - cm_final/cm_initial)*100:.1f}%")
    print(f"  Water conserved:  {engine.state.total_water():.2f} kg")


def demo_gymnasium_env():
    """4. Gymnasium environment with random agent."""
    divider("Demo 4: Random Agent Episode (10 s)")

    try:
        from habitat_sim.environment.habitat_env import HabitatEnv
    except ImportError:
        print("  Skipped — gymnasium not installed.")
        print("  Install with: pip install gymnasium")
        return

    cfg = reference_config()
    cfg.motor = MotorConfig(profile="off")
    cfg.simulation = SimulationConfig(dt=0.01, duration=10.0, control_dt=0.1)

    env = HabitatEnv(config=cfg)
    obs, _ = env.reset(seed=42)

    print(f"  Observation dim:  {len(obs)}")
    print(f"  Action dim:       {env.action_space.shape[0]}")

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

    print(f"  Steps:            {steps}")
    print(f"  Total reward:     {total_reward:.4f}")
    print(f"  Final nutation:   {info['nutation_angle_deg']:.4f} deg")

    env.close()


if __name__ == "__main__":
    print("Habitat Sim — Quick Demo")
    print(f"NumPy: {np.__version__}")

    demo_torque_free()
    demo_imbalance()
    demo_tank_correction()
    demo_gymnasium_env()

    divider("All demos complete")
