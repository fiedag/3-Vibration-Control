# Rotating Space Habitat Simulation

Dynamics simulation and reinforcement learning control system for a rotating space habitat operating in microgravity — a collaboration between a structural engineer specialising in spacecraft dynamics and vibration control, and an AI engineering agent.

---

## Background

The habitat is a large rotating structure (cylinder, ring, or toroid) whose spin generates artificial gravity for crew living on the inner wall. As crew and cargo move around the structure, mass imbalances develop and generate **precession**, **nutation**, and **vibration** — phenomena that must be actively suppressed to keep the habitat stable and comfortable.

This software simulates that physics and trains a neural network control system to counteract imbalance in real time by pumping water between rim tanks.

---

## Design

### Physical actuators

A set of water tanks distributed around the habitat rim, connected via manifolds. The control system commands valve flow rates between adjacent tanks to redistribute mass and counteract imbalance. The current design uses **36 rim tanks** (12 angular × 3 axial stations) with 3 central manifolds for axial equalisation.

### Control algorithm

**SAC (Soft Actor-Critic)** operating on a standard **MLP (Multi-Layer Perceptron)** with two hidden layers of 256 units. The agent observes accelerometer readings and mass estimates, and outputs continuous valve commands.

### Sensors

- **6 three-axis accelerometers** (2 per axial station, opposed at 0° and 180°) — capture centripetal acceleration and vibration signatures
- **Mass tracker** — noisy, rate-limited estimates of crew/cargo mass in each of the 36 habitat sectors

### Complexity levels

| Level | Description | Status |
|---|---|---|
| **Level 1** | Rigid body — no vibration modes | Implemented |
| **Level 2** | 1–2 vibration modes for critical speed analysis | Planned |
| **Level 3** | Full 20-mode flexible body model | Planned |

### Technology

Written in **Python** (VS Code), with the expectation that performance-critical elements will eventually be rewritten in **C++**. All experiments are stored in a **SQLite database** for later reporting and analysis.

---

## Quick Start

### 1. Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows
```

### 2. Install the package

```bash
# Core simulation only (numpy, scipy)
pip install -e .

# With RL training support
pip install -e ".[rl]"

# Everything (RL + database + visualization + dev tools)
pip install -e ".[all]"
```

> **Windows note:** Before installing `torch`, enable long paths to avoid a path-length error during install:
> ```powershell
> # Run as Administrator
> Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1
> ```
> Then restart your terminal.

### 3. Run the tests

```bash
pytest
```

All 113 tests should pass.

### 4. Run the demo

```bash
python scripts/quick_sim.py
```

Runs four demos: torque-free conservation check, mass imbalance whirl, tank correction, and a random Gymnasium agent episode.

### 5. Train an agent

```bash
habitat-train --timesteps 500000 --n-envs 4 --log-dir ./runs/my_run
```

Record training telemetry to a database:

```bash
habitat-train --timesteps 500000 --db habitat.db --experiment-name my_run
```

Analyse results:

```bash
python -m habitat_sim.scripts.analyse_experiment --db habitat.db --experiment-id 1
```

---

## Project Structure

```
habitat_sim/
├── config.py              Configuration dataclasses (all phases)
├── core/
│   ├── quaternion.py      Quaternion math
│   ├── inertia.py         Inertia tensor computation
│   └── integrator.py      RK4 integrator
├── geometry/
│   ├── base.py            Abstract geometry interface
│   ├── cylinder.py        Cylinder + ring geometry + factory
│   └── toroid.py          Thin-walled toroidal shell geometry
├── dynamics/
│   ├── base.py            Abstract dynamics interface
│   └── rigid_body.py      Level 1: Euler equations + tank ODEs
├── actuators/
│   ├── motor.py           Spin motor with torque profiles
│   └── tank_system.py     Rim tanks + manifolds
├── sensors/
│   ├── accelerometer.py   3-axis accelerometer model
│   ├── mass_tracker.py    Noisy sector mass estimator
│   └── sensor_suite.py    Bundles sensors → observation vector
├── disturbances/
│   ├── mass_schedule.py   Prescribed crew/cargo movement
│   ├── scenario.py        Combines disturbance sources + factory
│   └── stochastic.py      Poisson crew movement + micro-impacts
├── simulation/
│   ├── state.py           Flat state vector with named views
│   ├── engine.py          Simulation orchestrator
│   └── monitors.py        Conservation law checks
├── environment/
│   └── habitat_env.py     Gymnasium environment for RL
├── control/
│   ├── sac_agent.py       SAC construction + vectorised env helpers
│   └── training.py        Training loop, curriculum, evaluation
├── database/
│   ├── schema.py          SQLAlchemy 2.0 ORM (experiments/episodes/timesteps)
│   ├── recorder.py        ExperimentRecorder context manager + SB3 callback
│   └── queries.py         Read-side helpers (reward curves, nutation, etc.)
└── scripts/
    ├── train_agent.py         CLI: habitat-train
    ├── run_simulation.py      CLI: habitat-run
    └── analyse_experiment.py  CLI: plot training results
```

---

## Key Numbers

| Quantity | Default value |
|---|---|
| State dimension (Level 1) | 46 (quaternion 4 + omega 3 + tanks 36 + manifolds 3) |
| Observation dimension | 93 (accel 18 + sectors 36 + tanks 36 + manifolds 3) |
| Action dimension | 36 (valve commands) |
| Physics timestep | 0.01 s (100 Hz) |
| Control timestep | 0.1 s (10 Hz) |
| Sectors | 12 angular × 3 axial = 36 |
| Rim tanks | 12 per station × 3 stations = 36 |
| Manifolds | 3 (one per axial station) |
| Accelerometers | 6 (2 per station, opposed) |
| SAC network | [256, 256] MLP, ReLU |
| Curriculum stages | 4 (0 → 50 → 150 → 200 kg imbalance) |

---

## Configuration

All parameters live in `habitat_sim/config.py` as dataclasses. The top-level `ExperimentConfig` bundles all sub-configs and is JSON-serialisable.

```python
from habitat_sim.config import reference_config

cfg = reference_config()          # sensible defaults
cfg.rl.total_timesteps = 250_000  # override fields directly
cfg.stochastic.poisson_crew = True

print(cfg.to_json())              # serialise to JSON string
```

### Sub-configs

| Config | Purpose |
|---|---|
| `HabitatConfig` | Geometry shape, radius, wall density |
| `SectorConfig` | `n_angular × n_axial` sector grid |
| `TankConfig` | Tank count, capacity, water mass, flow rates |
| `MotorConfig` | Spin motor torque profile |
| `SensorConfig` | Accelerometer count and noise levels |
| `SimulationConfig` | `dt`, `control_dt`, episode duration |
| `RLConfig` | SAC hyperparameters, curriculum toggle |
| `StochasticConfig` | Poisson crew and micro-impact disturbances |

---

## Reinforcement Learning

### Training

```python
from habitat_sim.config import reference_config
from habitat_sim.control.training import run_training

cfg = reference_config()
cfg.rl.total_timesteps = 500_000
cfg.rl.n_envs = 4
model = run_training(cfg)
```

**Curriculum learning** is on by default (`cfg.rl.curriculum = True`). It ramps disturbance difficulty over four stages so the agent learns on easy problems before harder ones:

| Stage | Timestep fraction | Imbalance mass |
|---|---|---|
| 0 | 0–25% | 0 kg (clean) |
| 1 | 25–50% | 50 kg |
| 2 | 50–75% | 150 kg |
| 3 | 75–100% | 200 kg |

### Evaluation

```python
from habitat_sim.control.training import evaluate_agent

results = evaluate_agent("./runs/best_model/best_model.zip", cfg, n_episodes=10)
print(f"Mean reward: {results['mean_reward']:.2f}")
print(f"Mean nutation: {results['mean_nutation_deg']:.3f}°")
```

---

## Telemetry Database

All training runs are optionally recorded to a SQLite database with three tables:

| Table | Content |
|---|---|
| `experiments` | One row per training run — name, config JSON, timestamp |
| `episodes` | One row per episode — total reward, final nutation, CM offset |
| `timesteps` | One row per control step — ω, CM offset, water, energy, reward |

```python
from habitat_sim.database.recorder import ExperimentRecorder
from habitat_sim.control.training import run_training

with ExperimentRecorder("habitat.db", "experiment_1", cfg) as rec:
    run_training(cfg, recorder=rec)
```

Query results:

```python
from habitat_sim.database.queries import get_reward_curve, list_experiments

exps = list_experiments("habitat.db")
curve = get_reward_curve("habitat.db", experiment_id=1)
```

---

## Stochastic Disturbances

Two stochastic disturbance models simulate realistic crew activity and structural loading:

- **Poisson crew movement** — each crew member transitions between sectors at a Poisson rate; moves are smooth linear interpolations preserving total mass
- **Micro-impacts** — brief transient mass perturbations (micrometeorite impacts, structural loads) sampled from a Poisson process

```python
from habitat_sim.config import StochasticConfig
from habitat_sim.disturbances.scenario import build_scenario_from_stochastic_config

cfg.stochastic = StochasticConfig(
    poisson_crew=True,
    n_crew=6,
    mass_per_person=80.0,   # kg
    lambda_rate=0.01,       # sector transitions per second per crew member
    transfer_duration=30.0, # s — smooth transition window
)

scenario = build_scenario_from_stochastic_config(cfg.stochastic, n_sectors=36, seed=42)
masses = scenario.get_sector_masses(t=100.0)  # always sums to 480 kg
```

---

## Toroid Geometry

Switch to a toroidal habitat shape by changing the config. The state vector, sensors, and dynamics all scale automatically.

```python
from habitat_sim.config import SectorConfig, reference_config
from habitat_sim.simulation.engine import SimulationEngine
from dataclasses import replace
import numpy as np

cfg = reference_config()
cfg.habitat = replace(cfg.habitat, shape="toroid", minor_radius=5.0)
cfg.sectors = SectorConfig(n_angular=12, n_axial=1)   # no axial extent
cfg.tanks = replace(cfg.tanks, n_tanks_per_station=12, n_stations=1)
cfg.sensors = replace(cfg.sensors, n_accelerometers=2)

engine = SimulationEngine(cfg)
obs = engine.reset(seed=0)
obs, info = engine.step(np.zeros(cfg.tanks.n_tanks_total))
```

Inertia formulae for a thin-walled toroid (mass m, major radius R, minor radius r):

```
I_zz = m(2R² + 3r²) / 2    # spin axis
I_xx = I_yy = m(2R² + 5r²) / 4
```

---

## Implementation Status

- [x] Phase 1: Physics core (quaternion, inertia, Euler equations, RK4, spin motor) — 24 tests
- [x] Phase 2: Disturbances and tanks (mass schedules, hybrid manifold, scenarios) — 22 tests
- [x] Phase 3: Sensors and environment (accelerometers, mass tracker, Gymnasium) — 27 tests
- [x] Phase 4: RL training (SAC agent, curriculum, training loop, CLI) — 12 tests
- [x] Phase 5: Database and analysis (SQLite, experiment recording, plotting CLI) — 12 tests
- [x] Phase 6: Stochastic disturbances + toroid geometry — 18 tests

**Total: 113 tests, all passing.**

---

## Documentation

- `IMPLEMENTATION_GUIDE.md` — step-by-step reconstruction guide for Phases 4–6, including all file contents, design decisions, and engine fixes
- `docs/level1_rigid_body_dynamics.md` — full physics derivation
- `docs/code_architecture.md` — software architecture and design decisions
