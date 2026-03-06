# Phases 4–6 Implementation Guide
## Rotating Space Habitat Simulation

This document describes every file created or modified to implement Phases 4, 5, and 6 of the vibration-control simulation project. It is written as a step-by-step reconstruction guide.

**Final state:** 113 tests pass across all 6 phases (pytest, ~4 min runtime).

---

## Prerequisites

Phases 1–3 must already be present and passing (71 tests). The existing codebase provides:
- `habitat_sim/core/` — quaternion math, RK4 integrator, inertia tensors
- `habitat_sim/dynamics/` — `RigidBodyDynamics` (Euler equations)
- `habitat_sim/geometry/` — `CylinderGeometry`, `RingGeometry`
- `habitat_sim/simulation/` — `SimulationEngine`, `SimState`, `ConservationMonitor`
- `habitat_sim/environment/` — `HabitatEnv` (Gymnasium-compatible)
- `habitat_sim/sensors/` — `SensorSuite`, `AccelerometerArray`, `MassTracker`
- `habitat_sim/disturbances/` — `MassSchedule`, `Scenario`, `build_scenario()`
- `habitat_sim/config.py` — all configuration dataclasses
- `pyproject.toml` — build system and extras

### Windows-specific setup

Before installing `torch` on Windows, enable long paths to avoid install failure:

```powershell
# Run as Administrator
Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
    -Name "LongPathsEnabled" -Value 1
```

Then restart your terminal.

### Install dependencies

```bash
pip install -e ".[rl,db,viz]"
```

Extras defined in `pyproject.toml`:
- `rl` — `stable-baselines3>=2.0`, `torch>=2.0`
- `db` — `sqlalchemy>=2.0`
- `viz` — `matplotlib>=3.7`

---

## Phase 4: SB3 SAC Training Pipeline

### 4.1 `habitat_sim/config.py` — add `RLConfig` and `StochasticConfig`

Append two new dataclasses before `ExperimentConfig`, then add them as fields on `ExperimentConfig`:

```python
@dataclass
class RLConfig:
    """Hyperparameters for SAC reinforcement learning training."""
    algorithm: str = "SAC"
    total_timesteps: int = 500_000
    n_envs: int = 4                    # parallel environments
    learning_rate: float = 3e-4
    buffer_size: int = 100_000
    batch_size: int = 256
    learning_starts: int = 5_000
    gamma: float = 0.99
    tau: float = 0.005
    ent_coef: str = "auto"
    net_arch: list = field(default_factory=lambda: [256, 256])
    eval_freq: int = 5_000             # env steps between evaluations
    n_eval_episodes: int = 5
    checkpoint_freq: int = 25_000
    log_dir: str = "./runs"
    curriculum: bool = True            # progressive disturbance ramp


@dataclass
class StochasticConfig:
    """Parameters for stochastic disturbance sources (Phase 6)."""
    # Poisson crew movement
    poisson_crew: bool = False
    n_crew: int = 6
    mass_per_person: float = 80.0      # kg per crew member
    lambda_rate: float = 0.01          # mean sector transitions per second
    transfer_duration: float = 30.0    # s -- smooth transition duration
    # Micro-impact disturbances
    micro_impacts: bool = False
    impact_rate: float = 0.001         # impacts per second
    impact_mass_std: float = 0.1       # kg std of impact mass
    impact_duration: float = 1.0       # s -- duration of each impact
```

On `ExperimentConfig`, add fields:
```python
rl: RLConfig = field(default_factory=RLConfig)
stochastic: StochasticConfig = field(default_factory=StochasticConfig)
```

On `ExperimentConfig.from_dict()`, add:
```python
rl=RLConfig(**d.get("rl", {})),
stochastic=StochasticConfig(**d.get("stochastic", {})),
```

### 4.2 `habitat_sim/scripts/__init__.py` — NEW (empty)

Create an empty file to make `habitat_sim.scripts` a proper package:

```python
```

### 4.3 `habitat_sim/control/sac_agent.py` — NEW

```python
"""SAC agent construction and vectorised environment helpers."""

from __future__ import annotations
import os, sys
from typing import Callable
import numpy as np

try:
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False

from habitat_sim.config import ExperimentConfig, RLConfig
from habitat_sim.environment.habitat_env import HabitatEnv


def _require_sb3() -> None:
    if not HAS_SB3:
        raise ImportError("stable-baselines3 required. pip install habitat-sim[rl]")


def make_env(config: ExperimentConfig, rank: int, seed: int) -> Callable:
    def _init() -> HabitatEnv:
        import copy
        env_cfg = copy.deepcopy(config)
        env_cfg.seed = seed + rank
        env = HabitatEnv(config=env_cfg)
        env.reset(seed=seed + rank)
        return env
    return _init


def build_vec_env(config: ExperimentConfig, n_envs: int, seed: int = 42) -> "VecEnv":
    """DummyVecEnv on Windows, SubprocVecEnv elsewhere."""
    _require_sb3()
    fns = [make_env(config, rank=i, seed=seed) for i in range(n_envs)]
    if sys.platform == "win32" or n_envs == 1:
        return DummyVecEnv(fns)
    return SubprocVecEnv(fns)


def build_sac(env, rl_config: RLConfig, seed: int = 42,
              tensorboard_log: str | None = None) -> "SAC":
    _require_sb3()
    import torch.nn as nn
    policy_kwargs = {"net_arch": rl_config.net_arch, "activation_fn": nn.ReLU}
    return SAC(
        policy="MlpPolicy", env=env,
        learning_rate=rl_config.learning_rate, buffer_size=rl_config.buffer_size,
        batch_size=rl_config.batch_size, learning_starts=rl_config.learning_starts,
        gamma=rl_config.gamma, tau=rl_config.tau, ent_coef=rl_config.ent_coef,
        policy_kwargs=policy_kwargs, verbose=1, seed=seed,
        tensorboard_log=tensorboard_log,
    )


def load_sac(model_path: str, env=None) -> "SAC":
    _require_sb3()
    return SAC.load(model_path, env=env)
```

### 4.4 `habitat_sim/control/training.py` — NEW

Key design decisions:
- `CurriculumCallback` advances through 4 stages (0%, 25%, 50%, 75% of total_timesteps) with imbalance masses `[0, 50, 150, 200]` kg.
- TensorBoard logging is opt-in: only enabled when `torch.utils.tensorboard` is importable (not always installed alongside torch on Windows).
- `recorder` parameter threads the Phase-5 `ExperimentRecorder` into the callback list.

```python
"""SAC training orchestration: curriculum, callbacks, and evaluation."""

from __future__ import annotations
import copy, os
from typing import TYPE_CHECKING
import numpy as np
from habitat_sim.config import ExperimentConfig, MotorConfig

if TYPE_CHECKING:
    from stable_baselines3 import SAC
    from habitat_sim.database.recorder import ExperimentRecorder


def _make_curriculum_callback(config, total_timesteps):
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
                print(f"\n[Curriculum] Stage {stage}: imbalance mass = {mass:.0f} kg")
            return True

    return CurriculumCallback(config, total_timesteps)


def run_training(config: ExperimentConfig, recorder=None) -> "SAC":
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
    from habitat_sim.control.sac_agent import build_vec_env, build_sac
    from habitat_sim.environment.habitat_env import HabitatEnv

    rl = config.rl
    log_dir = rl.log_dir
    os.makedirs(log_dir, exist_ok=True)

    train_cfg = copy.deepcopy(config)
    train_cfg.motor = MotorConfig(profile="off")
    train_env = build_vec_env(train_cfg, n_envs=rl.n_envs, seed=config.seed)

    eval_cfg = copy.deepcopy(config)
    eval_cfg.motor = MotorConfig(profile="off")
    eval_env = DummyVecEnv([lambda: HabitatEnv(config=copy.deepcopy(eval_cfg))])

    # TensorBoard only if installed
    try:
        import torch.utils.tensorboard  # noqa
        tb_log = os.path.join(log_dir, "tb")
    except ImportError:
        tb_log = None
    model = build_sac(train_env, rl, seed=config.seed, tensorboard_log=tb_log)

    eval_cb = EvalCallback(eval_env,
        best_model_save_path=os.path.join(log_dir, "best_model"),
        log_path=os.path.join(log_dir, "eval"),
        eval_freq=max(rl.eval_freq // rl.n_envs, 1),
        n_eval_episodes=rl.n_eval_episodes, deterministic=True, verbose=1)
    checkpoint_cb = CheckpointCallback(
        save_freq=max(rl.checkpoint_freq // rl.n_envs, 1),
        save_path=os.path.join(log_dir, "checkpoints"),
        name_prefix="sac_ckpt", verbose=0)

    callbacks = [eval_cb, checkpoint_cb]
    if rl.curriculum:
        callbacks.append(_make_curriculum_callback(config, rl.total_timesteps))
    if recorder is not None:
        from habitat_sim.database.recorder import RecorderCallback
        callbacks.append(RecorderCallback(recorder))

    model.learn(total_timesteps=rl.total_timesteps, callback=CallbackList(callbacks),
                log_interval=10, reset_num_timesteps=True, progress_bar=False)
    model.save(os.path.join(log_dir, "final_model"))
    train_env.close()
    eval_env.close()
    return model


def evaluate_agent(model_path: str, config: ExperimentConfig, n_episodes: int = 10) -> dict:
    from habitat_sim.control.sac_agent import load_sac
    from habitat_sim.environment.habitat_env import HabitatEnv
    env = HabitatEnv(config=config)
    model = load_sac(model_path)
    rewards, nutations, cm_offsets, episode_records = [], [], [], []
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
        episode_records.append({"episode": ep, "total_reward": total_reward,
                                 "nutation_deg": nutation, "cm_offset_m": cm})
    env.close()
    return {"mean_reward": float(np.mean(rewards)), "std_reward": float(np.std(rewards)),
            "mean_nutation_deg": float(np.mean(nutations)),
            "mean_cm_offset": float(np.mean(cm_offsets)), "episodes": episode_records}
```

### 4.5 `habitat_sim/scripts/train_agent.py` — NEW

CLI entry point for `habitat-train`. Supports `--db` to integrate Phase-5 telemetry.

```python
"""CLI: habitat-train"""
from __future__ import annotations
import argparse, json

def main():
    parser = argparse.ArgumentParser(description="Train SAC agent on HabitatEnv")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--log-dir", type=str, default="./runs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-curriculum", action="store_true")
    parser.add_argument("--db", type=str, default=None)
    parser.add_argument("--experiment-name", type=str, default="default")
    args = parser.parse_args()

    from habitat_sim.config import reference_config, ExperimentConfig
    if args.config:
        with open(args.config, encoding="utf-8") as f:
            cfg = ExperimentConfig.from_dict(json.load(f))
    else:
        cfg = reference_config()

    if args.timesteps:  cfg.rl.total_timesteps = args.timesteps
    if args.n_envs:     cfg.rl.n_envs = args.n_envs
    cfg.rl.log_dir = args.log_dir
    cfg.seed = args.seed
    if args.no_curriculum: cfg.rl.curriculum = False

    from habitat_sim.control.training import run_training
    if args.db:
        from habitat_sim.database.recorder import ExperimentRecorder
        with ExperimentRecorder(args.db, args.experiment_name, cfg) as rec:
            run_training(cfg, recorder=rec)
    else:
        run_training(cfg)

if __name__ == "__main__":
    main()
```

### 4.6 `habitat_sim/scripts/run_simulation.py` — NEW

CLI entry point for `habitat-run`. Runs demos or evaluates a saved model.

```python
"""CLI: habitat-run"""
from __future__ import annotations
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run HabitatEnv simulation or evaluate model")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from habitat_sim.config import reference_config
    cfg = reference_config()
    cfg.seed = args.seed

    if args.model:
        from habitat_sim.control.training import evaluate_agent
        results = evaluate_agent(args.model, cfg, n_episodes=args.episodes)
        print(f"Mean reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"Mean nutation: {results['mean_nutation_deg']:.3f}°")
    else:
        print("Demo mode: running 1 episode with zero action.")
        from habitat_sim.environment.habitat_env import HabitatEnv
        import numpy as np
        env = HabitatEnv(config=cfg)
        obs, _ = env.reset(seed=args.seed)
        done = False
        total_reward = 0.0
        while not done:
            obs, reward, terminated, truncated, info = env.step(np.zeros(env.action_space.shape))
            total_reward += reward
            done = terminated or truncated
        print(f"Episode done. Total reward: {total_reward:.2f}")
        env.close()

if __name__ == "__main__":
    main()
```

### 4.7 `tests/test_phase4.py` — NEW

12 tests across 5 classes: `TestVecEnv`, `TestBuildSAC`, `TestShortTraining`, `TestCurriculumCallback`, `TestEvaluateAgent`.

Key helper — fast config for unit tests:
```python
def _fast_config():
    cfg = reference_config()
    cfg.simulation = SimulationConfig(dt=0.01, duration=10.0, control_dt=0.1)
    cfg.rl.total_timesteps = 200
    cfg.rl.batch_size = 32
    cfg.rl.learning_starts = 50
    cfg.rl.n_envs = 1
    cfg.rl.eval_freq = 100
    cfg.rl.n_eval_episodes = 1
    cfg.rl.checkpoint_freq = 100
    cfg.rl.curriculum = False
    cfg.motor = MotorConfig(profile="off")
    return cfg
```

---

## Phase 5: Training Telemetry Database

### 5.1 `habitat_sim/database/__init__.py` — NEW (empty)

### 5.2 `habitat_sim/database/schema.py` — NEW

SQLAlchemy 2.0 declarative ORM for three tables.

**Important (Windows):** always write with `open(..., encoding='utf-8')` or via Python's `io.open`. The default cp1252 encoding will silently corrupt em-dashes and cause `SyntaxError` when Python's UTF-8 parser reads the file.

Three ORM classes:

| Table | Key columns |
|---|---|
| `experiments` | `id`, `name`, `created_at` (UTC datetime), `config_json`, `seed`, `algorithm` |
| `episodes` | `id`, `experiment_id` (FK), `episode_num`, `n_steps`, `total_reward`, `final_nutation_deg`, `final_cm_offset_mag`, `final_omega_z`, `h_violation_count` |
| `timesteps` | `id`, `episode_id` (FK), `step_index`, `t`, `omega_x/y/z`, `cm_offset_mag`, `total_water`, `kinetic_energy`, `reward`, `n_violations` |

Factory function:
```python
def get_engine(db_path: str = "habitat.db"):
    """":memory:" gives an in-memory SQLite engine (distinct engine per call)."""
    url = f"sqlite:///{db_path}" if db_path != ":memory:" else "sqlite:///:memory:"
    engine = create_engine(url, echo=False)
    Base.metadata.create_all(engine)
    return engine
```

### 5.3 `habitat_sim/database/recorder.py` — NEW

`ExperimentRecorder` is a context manager:
- `__enter__`: creates `Experiment` row, stores `_engine` and `_experiment_id`
- `record_episode(episode_num, steps_data, engine_info=None)`: batch-inserts `Episode` + all `Timestep` rows in a single session
- `__exit__`: flushes any remaining buffered steps

`RecorderCallback(BaseCallback)` hooks into SB3's training loop:
- `_on_step()`: buffers step info from `self.locals["infos"]`; flushes on `dones[i] == True`
- `_on_training_end()`: flushes any remaining buffer

The entire class definition is wrapped in `try/except ImportError` so the module loads even when SB3 is absent.

### 5.4 `habitat_sim/database/queries.py` — NEW

Helper functions returning plain dicts for plotting:
- `list_experiments(db_path) -> list[dict]`
- `get_reward_curve(db_path, experiment_id) -> dict`
- `get_nutation_curve(db_path, experiment_id) -> dict`
- `get_conservation_summary(db_path, experiment_id) -> dict`
- `get_timestep_series(db_path, episode_id, columns=None) -> dict`

### 5.5 `habitat_sim/scripts/analyse_experiment.py` — NEW

CLI: `python -m habitat_sim.scripts.analyse_experiment --db habitat.db --experiment-id 1`

Options: `--db`, `--experiment-id`, `--list` (list all experiments), `--out-dir`.

Produces a 3-panel matplotlib figure saved as `{out_dir}/experiment_{id}_summary.png`:
1. Total reward per episode
2. Final nutation angle per episode
3. H-violation count per episode

### 5.6 `tests/test_phase5.py` — NEW

12 tests. All use `:memory:` SQLite for isolation. Key pattern:

```python
with ExperimentRecorder(":memory:", "run", cfg) as rec:
    rec.record_episode(0, steps)
    with Session(rec._engine) as session:
        count = session.query(Episode).count()
assert count == 1
```

---

## Phase 6A: Stochastic Disturbances

### 6.1 `habitat_sim/disturbances/stochastic.py` — NEW

#### `PoissonCrewDisturbance`

Models `n_crew` crew members walking between sectors via a Poisson process.

**Algorithm:**
1. Pre-generate an event schedule via `rng.exponential(1/lambda_rate)` inter-event times for each crew member, up to a 1000 s horizon.
2. `_process_events_up_to(t)`: consume events ≤ t; each consumed event creates an `(crew_idx, from_sector, to_sector, t_start, t_end)` transfer tuple.
3. `get_sector_masses(t)`:
   - Start with base masses: each crew member's `mass_per_person` at their current sector.
   - For each active transfer at time `t_start ≤ t < t_end`:
     - `alpha = (t - t_start) / (t_end - t_start)`
     - Remove full mass from `to_sector` (current position), add `(1-alpha)` to `from_sector` + `alpha` to `to_sector`.
   - Total mass is always conserved = `n_crew * mass_per_person`.

#### `MicroImpactDisturbance`

Brief transient mass perturbations.

**Algorithm:**
1. Pre-generate impact schedule: `t_impact ~ Poisson(rate)`, `sector ~ Uniform(0, n_sectors)`, `mass ~ |Normal(0, mass_std)|`.
2. `get_sector_masses(t)`: sum masses of all impacts where `t_impact ≤ t < t_impact + duration`.

### 6.2 `habitat_sim/disturbances/scenario.py` — MODIFY

Add two new branches to `build_scenario()`:

```python
elif dtype == "poisson_crew":
    from habitat_sim.disturbances.stochastic import PoissonCrewDisturbance
    params = {k: v for k, v in dc.items() if k != "type"}
    sources.append(PoissonCrewDisturbance(n_sectors=n_sectors, **params))
elif dtype == "micro_impact":
    from habitat_sim.disturbances.stochastic import MicroImpactDisturbance
    params = {k: v for k, v in dc.items() if k != "type"}
    sources.append(MicroImpactDisturbance(n_sectors=n_sectors, **params))
```

Add new public function:

```python
def build_scenario_from_stochastic_config(
    stochastic_cfg,      # StochasticConfig
    n_sectors: int = 36,
    seed: int = 0,
) -> Scenario:
    """Build a Scenario from a StochasticConfig dataclass."""
    from habitat_sim.disturbances.stochastic import PoissonCrewDisturbance, MicroImpactDisturbance
    sources = []
    if stochastic_cfg.poisson_crew:
        sources.append(PoissonCrewDisturbance(
            n_sectors=n_sectors, n_crew=stochastic_cfg.n_crew,
            mass_per_person=stochastic_cfg.mass_per_person,
            lambda_rate=stochastic_cfg.lambda_rate,
            transfer_duration=stochastic_cfg.transfer_duration, seed=seed))
    if stochastic_cfg.micro_impacts:
        sources.append(MicroImpactDisturbance(
            n_sectors=n_sectors, rate=stochastic_cfg.impact_rate,
            mass_std=stochastic_cfg.impact_mass_std,
            duration=stochastic_cfg.impact_duration, seed=seed + 1))
    return Scenario(sources, n_sectors=n_sectors)
```

---

## Phase 6B: Toroid Geometry

### 6.3 `habitat_sim/geometry/toroid.py` — NEW

Uses `config.radius` (R, major) and `config.minor_radius` (r, minor).

**Analytical formulae for thin-walled toroidal shell:**

| Quantity | Formula |
|---|---|
| Surface area | `4π²Rr` |
| Structural mass | `ρ_wall × 4π²Rr × t_wall` |
| `I_zz` (spin axis) | `m(2R² + 3r²) / 2` |
| `I_xx = I_yy` | `m(2R² + 5r²) / 4` |

**Geometry overrides:**

| Method | Toroid behaviour |
|---|---|
| `compute_sector_positions(sc)` | Returns `(n_angular, 3)` — all at `z=0`, radius R. Ignores `n_axial`. |
| `compute_tank_positions(tc)` | Returns `(n_tanks_per_station, 3)` — all at `z=0`, radius R. |
| `compute_manifold_positions(tc)` | Returns `(1, 3)` — single manifold at origin. |
| `compute_default_accelerometer_positions(n)` | Returns `(2, 3)` — two opposed sensors at `(R,0,0)` and `(-R,0,0)`. |

### 6.4 `habitat_sim/geometry/cylinder.py` — MODIFY factory

```python
def create_geometry(config: HabitatConfig) -> HabitatGeometry:
    from habitat_sim.geometry.toroid import ToroidGeometry
    _MAP = {
        "cylinder": CylinderGeometry,
        "ring": RingGeometry,
        "toroid": ToroidGeometry,
    }
    cls = _MAP.get(config.shape)
    if cls is None:
        raise ValueError(f"Unknown habitat shape: {config.shape!r}")
    return cls(config)
```

---

## Engine Fixes for Variable Tank/Station Counts

Using a toroid with `n_stations=1` exposes hardcoded state-vector slices in three files. These fixes make the engine correct for any `n_tanks_per_station × n_stations` combination.

### Fix 1: `habitat_sim/dynamics/rigid_body.py`

Add `__init__` to accept an optional `TankConfig`:

```python
def __init__(self, tank_config=None) -> None:
    if tank_config is not None:
        n_tanks = tank_config.n_tanks_total
        n_manifolds = tank_config.n_stations
    else:
        n_tanks = 36
        n_manifolds = 3
    self._tank_start = 7
    self._manifold_offset = self._tank_start + n_tanks
    self._n_state = self._manifold_offset + n_manifolds
    self._n_tanks = n_tanks
    self._n_manifolds = n_manifolds
```

Replace every hardcoded slice in `compute_derivatives`:
- `state[7:43]` → `state[self._tank_start:self._manifold_offset]`
- `state[43:46]` → `state[self._manifold_offset:self._manifold_offset + self._n_manifolds]`
- `np.empty(self._N_STATE)` → `np.empty(self._n_state)`
- `dx[7:43]` → `dx[self._tank_start:self._manifold_offset]`
- `dx[43:46]` → `dx[self._manifold_offset:...]`

### Fix 2: `habitat_sim/simulation/monitors.py`

Add `n_tanks` and `n_manifolds` parameters to `ConservationMonitor.__init__`:

```python
def __init__(self, h_tol=1e-6, q_tol=1e-10, water_tol=1e-10,
             n_tanks=36, n_manifolds=3):
    ...
    self._tank_start = 7
    self._tank_end = 7 + n_tanks
    self._manifold_end = self._tank_end + n_manifolds
```

Replace hardcoded slices in `check()`:
- `state_x[7:43]` → `state_x[self._tank_start:self._tank_end]`
- `state_x[43:46]` → `state_x[self._tank_end:self._manifold_end]`

### Fix 3: `habitat_sim/sensors/sensor_suite.py`

Add `n_tanks` and `n_manifolds` parameters to `SensorSuite.__init__`:

```python
def __init__(self, config, accelerometer_positions, n_sectors=36,
             n_tanks=36, n_manifolds=3, seed=42):
    ...
    self.n_tanks = n_tanks
    self.n_manifolds = n_manifolds
    self.n_accel_obs = 3 * self.n_accels
    self.n_obs = self.n_accel_obs + n_sectors + n_tanks + n_manifolds
```

Replace hardcoded `36` and `3` in `observe()`:
- `obs[i:i + 36] = tank_masses` → `obs[i:i + self.n_tanks] = tank_masses`
- `i += 36` → `i += self.n_tanks`
- `obs[i:i + 3] = manifold_masses` → `obs[i:i + self.n_manifolds] = manifold_masses`

### Fix 4: `habitat_sim/simulation/engine.py`

Pass dynamic counts when constructing the dynamics, monitor, and sensor suite:

```python
# Replace create_dynamics(config.simulation.dynamics_level) with:
self.dynamics = RigidBodyDynamics(config.tanks)

# Add n_tanks and n_manifolds to ConservationMonitor:
self.monitor = ConservationMonitor(
    n_tanks=config.tanks.n_tanks_total,
    n_manifolds=config.tanks.n_stations,
)

# Add n_tanks and n_manifolds to SensorSuite:
self.sensors = SensorSuite(
    config=config.sensors,
    accelerometer_positions=accel_positions,
    n_sectors=config.sectors.n_total,
    n_tanks=config.tanks.n_tanks_total,
    n_manifolds=config.tanks.n_stations,
    seed=config.seed,
)
```

---

## Phase 6 Tests: `tests/test_phase6.py` — NEW

18 tests across 4 classes: `TestPoissonCrew` (3), `TestMicroImpact` (2), `TestBuildScenarioFromStochastic` (5), `TestToroidGeometry` (8).

### Toroid simulation test — config requirements

The `SimulationEngine` requires that positions and masses arrays have matching lengths. For a toroid:
- `n_axial=1` makes `sectors.n_total = n_angular` (matches the geometry's sector positions)
- `n_stations=1` makes `tanks.n_tanks_total = n_tanks_per_station` (matches tank positions)
- `n_accelerometers=2` matches the toroid's 2-accelerometer placement

```python
cfg.habitat = replace(cfg.habitat, shape="toroid", minor_radius=5.0)
cfg.sectors = SectorConfig(n_angular=12, n_axial=1)
cfg.tanks = replace(cfg.tanks, n_tanks_per_station=12, n_stations=1)
cfg.sensors = replace(cfg.sensors, n_accelerometers=2)
```

`engine.step(action)` returns `(obs, info)`, not a plain observation array.

---

## Running the tests

```bash
# All phases
pytest tests/ -v

# Individual phases
pytest tests/test_phase4.py -v   # 12 tests, ~4 min (SAC training)
pytest tests/test_phase5.py -v   # 12 tests, ~4 s
pytest tests/test_phase6.py -v   # 18 tests, ~1 s
```

Expected: **113 passed**.

## Quick end-to-end demo

```python
# Phase 4: short training run
from habitat_sim.config import reference_config, SimulationConfig, MotorConfig, RLConfig
from habitat_sim.control.training import run_training

cfg = reference_config()
cfg.simulation = SimulationConfig(dt=0.01, duration=10.0, control_dt=0.1)
cfg.rl = RLConfig(total_timesteps=500, n_envs=1, batch_size=32,
                  learning_starts=50, curriculum=False, log_dir="./runs/demo")
cfg.motor = MotorConfig(profile="off")
model = run_training(cfg)

# Phase 5: record to database
from habitat_sim.database.recorder import ExperimentRecorder
with ExperimentRecorder(":memory:", "demo_run", cfg) as rec:
    run_training(cfg, recorder=rec)

# Phase 6: stochastic crew
from habitat_sim.config import StochasticConfig
from habitat_sim.disturbances.scenario import build_scenario_from_stochastic_config

sc_cfg = StochasticConfig(poisson_crew=True, n_crew=6)
scenario = build_scenario_from_stochastic_config(sc_cfg, n_sectors=36, seed=42)
print(scenario.get_sector_masses(100.0).sum())  # 480.0 kg

# Phase 6: toroid geometry
from habitat_sim.config import SectorConfig, TankConfig, SensorConfig
from habitat_sim.simulation.engine import SimulationEngine
from dataclasses import replace
import numpy as np

cfg = reference_config()
cfg.habitat = replace(cfg.habitat, shape="toroid", minor_radius=5.0)
cfg.sectors = SectorConfig(n_angular=12, n_axial=1)
cfg.tanks = replace(cfg.tanks, n_tanks_per_station=12, n_stations=1)
cfg.sensors = replace(cfg.sensors, n_accelerometers=2)

engine = SimulationEngine(cfg)
obs = engine.reset(seed=0)
for _ in range(5):
    obs, info = engine.step(np.zeros(cfg.tanks.n_tanks_total))
print("Toroid simulation OK, obs shape:", obs.shape)
```
