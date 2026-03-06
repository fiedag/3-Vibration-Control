# Code Architecture: Rotating Space Habitat Simulation

## 1. Design Principles

**Separation of concerns.** The physics engine knows nothing about RL. The RL environment wraps the physics engine and translates between physics state and agent observations/actions. The database layer is orthogonal to both.

**Configuration-driven.** Every physical parameter (geometry, masses, sensor placement, tank layout, motor profiles) is specified in a configuration object, not hardcoded. This lets us sweep parameters without touching code.

**Level-agnostic core.** The integrator, coordinate transforms, and state management are shared across Levels 1–3. The dynamics model is swappable — Level 1 plugs in rigid body equations, Level 2 adds modal states, Level 3 extends to 20 modes. The interface between the integrator and the dynamics model is fixed.

**NumPy-first, C++-ready.** All physics computations use NumPy arrays with shapes that map directly to C arrays. No Python objects in the inner loop. When we migrate to C++, the physics core becomes a shared library called via ctypes or pybind11, and the Python wrapper stays identical.

---

## 2. Package Structure

```
habitat_sim/
├── __init__.py
├── config.py                  # Configuration dataclasses
├── core/
│   ├── __init__.py
│   ├── quaternion.py          # Quaternion math (multiply, normalize, to_rotation_matrix, to_euler)
│   ├── inertia.py             # Inertia tensor computation from geometry + point masses
│   └── integrator.py          # RK4 integrator with quaternion normalisation
├── geometry/
│   ├── __init__.py
│   ├── base.py                # Abstract HabitatGeometry class
│   ├── cylinder.py            # Cylinder (with end plates)
│   ├── ring.py                # Open cylinder
│   └── toroid.py              # Toroid
├── dynamics/
│   ├── __init__.py
│   ├── base.py                # Abstract DynamicsModel interface
│   ├── rigid_body.py          # Level 1: Euler equations with variable inertia
│   ├── modal_2.py             # Level 2: rigid body + 2 flex modes (future)
│   └── modal_20.py            # Level 3: rigid body + 20 flex modes (future)
├── actuators/
│   ├── __init__.py
│   ├── tank_system.py         # 36 rim tanks + 3 manifolds + hybrid plumbing
│   └── motor.py               # Spin motor with torque profiles
├── sensors/
│   ├── __init__.py
│   ├── accelerometer.py       # 3-axis accelerometer model with noise
│   └── mass_tracker.py        # Sector mass estimation with noise + update rate
├── disturbances/
│   ├── __init__.py
│   ├── mass_schedule.py       # Prescribed crew/cargo movement events
│   ├── stochastic.py          # Random walk, activity-based, correlated movement
│   └── scenario.py            # Scenario: combines multiple disturbance sources
├── simulation/
│   ├── __init__.py
│   ├── state.py               # SimState: full state vector + unpacking utilities
│   ├── engine.py              # SimulationEngine: ties everything together, steps forward
│   └── monitors.py            # Conservation law checks, diagnostics
├── environment/
│   ├── __init__.py
│   └── habitat_env.py         # Gymnasium environment wrapping SimulationEngine
├── control/
│   ├── __init__.py
│   ├── sac_agent.py           # SAC + MLP policy (wraps stable-baselines3 or clean impl)
│   └── training.py            # Training loop, hyperparameters, curriculum
├── database/
│   ├── __init__.py
│   ├── schema.py              # SQLAlchemy models (experiments, episodes, snapshots)
│   ├── recorder.py            # Records simulation data to DB during/after runs
│   └── queries.py             # Analysis queries (compare experiments, extract time series)
├── visualization/
│   ├── __init__.py
│   ├── plots.py               # Matplotlib: time series, phase plots, momentum vectors
│   ├── scene_3d.py            # 3D habitat rendering with momentum/spin axis vectors
│   └── dashboard.py           # Live or post-hoc dashboard (optional, later)
└── scripts/
    ├── run_simulation.py      # CLI: run a single simulation from config file
    ├── train_agent.py         # CLI: train SAC agent
    ├── evaluate_agent.py      # CLI: evaluate trained agent on scenarios
    └── analyse_experiment.py  # CLI: pull data from DB and generate reports
```

---

## 3. Core Abstractions

### 3.1 Configuration (`config.py`)

All parameters in one place, serialisable to JSON/YAML for experiment reproducibility.

```python
@dataclass
class HabitatConfig:
    shape: str                     # "cylinder", "ring", "toroid"
    radius: float                  # m (R for cylinder/ring, R_maj for toroid)
    length: float                  # m (L for cylinder/ring, unused for toroid)
    minor_radius: float            # m (r for toroid, unused otherwise)
    wall_thickness: float          # m
    wall_density: float            # kg/m³
    end_plate_thickness: float     # m (cylinder only)
    end_plate_density: float       # kg/m³ (cylinder only)

@dataclass
class SectorConfig:
    n_angular: int = 12
    n_axial: int = 3               # 1 for toroid

@dataclass
class TankConfig:
    n_tanks_per_station: int = 12
    n_stations: int = 3
    tank_capacity: float           # kg (m_tank_max per tank)
    total_water_mass: float        # kg (m_water_total)
    initial_distribution: str      # "uniform", "custom"
    q_circ_max: float              # kg/s (circumferential pump rate limit)
    q_axial_max: float             # kg/s (axial transfer rate limit)

@dataclass
class MotorConfig:
    profile: str                   # "constant", "ramp", "trapezoidal", "s_curve", "custom"
    max_torque: float              # N·m
    ramp_time: float               # s
    hold_time: float               # s (trapezoidal only)
    target_spin_rate: float        # rad/s

@dataclass
class SensorConfig:
    n_accelerometers: int = 6
    accelerometer_positions: np.ndarray  # (N_s, 3) body-frame positions
    accelerometer_noise_std: float       # m/s² per axis
    mass_tracker_noise_std: float        # kg per sector
    mass_tracker_update_rate: float      # Hz

@dataclass
class SimulationConfig:
    dt: float = 0.01               # s (integration time step)
    duration: float = 3600.0       # s (total simulation time)
    control_dt: float = 0.1        # s (RL agent decision interval)
    dynamics_level: int = 1        # 1, 2, or 3

@dataclass
class ExperimentConfig:
    habitat: HabitatConfig
    sectors: SectorConfig
    tanks: TankConfig
    motor: MotorConfig
    sensors: SensorConfig
    simulation: SimulationConfig
    disturbances: list              # list of disturbance configs
    seed: int = 42
```

### 3.2 Simulation State (`simulation/state.py`)

A flat NumPy array with named slices. No Python objects in the hot path.

```python
class SimState:
    """Manages the full state vector as a flat numpy array with named views."""

    # Index map (Level 1, 46 states):
    #   [0:4]    quaternion (q0, q1, q2, q3)
    #   [4:7]    angular velocity (omega_x, omega_y, omega_z)
    #   [7:43]   tank masses (36 tanks, row-major: station 1 angular 1..12, station 2, station 3)
    #   [43:46]  manifold masses (3 manifolds)

    def __init__(self, config: ExperimentConfig):
        self.n_rigid = 7
        self.n_tanks = config.tanks.n_tanks_per_station * config.tanks.n_stations  # 36
        self.n_manifolds = config.tanks.n_stations  # 3
        self.n_total = self.n_rigid + self.n_tanks + self.n_manifolds  # 46

        self.x = np.zeros(self.n_total)
        self.x[0] = 1.0  # quaternion scalar = 1 (identity rotation)

    # --- Named views (zero-copy slices) ---
    @property
    def quaternion(self) -> np.ndarray:      # shape (4,)
        return self.x[0:4]

    @property
    def omega(self) -> np.ndarray:           # shape (3,)
        return self.x[4:7]

    @property
    def tank_masses(self) -> np.ndarray:     # shape (36,) or reshaped to (3, 12)
        return self.x[7:43]

    @property
    def tank_masses_2d(self) -> np.ndarray:  # shape (3, 12) — [station, angular]
        return self.x[7:43].reshape(3, 12)

    @property
    def manifold_masses(self) -> np.ndarray: # shape (3,)
        return self.x[43:46]
```

### 3.3 Dynamics Model Interface (`dynamics/base.py`)

The integrator calls this interface. Level 1, 2, 3 each implement it.

```python
class DynamicsModel(ABC):
    """Interface for computing state derivatives."""

    @abstractmethod
    def compute_derivatives(
        self,
        t: float,
        state: np.ndarray,
        sector_masses: np.ndarray,       # (36,) current crew/cargo masses
        tank_valve_commands: np.ndarray,  # (36,) normalised valve commands [-1, +1]
        motor_torque: float,             # N·m about z_B
        config: ExperimentConfig
    ) -> np.ndarray:
        """Returns dx/dt as a flat array matching state dimensions."""
        ...

    @abstractmethod
    def state_dimension(self) -> int:
        """Total number of state variables."""
        ...
```

Level 1 implementation computes:
1. Inertia tensor from geometry + sector masses + tank masses + manifold masses
2. dI/dt from mass movement rates
3. Euler equation: dω/dt = I⁻¹ [τ_ext − dI/dt·ω − ω×(Iω)]
4. Quaternion kinematics: dq/dt = ½ Ω(ω)·q
5. Tank dynamics: dm_tank/dt from valve commands
6. Manifold dynamics: dm_manifold/dt from net valve flows + axial equalisation

### 3.4 Simulation Engine (`simulation/engine.py`)

Orchestrates one time step: queries disturbances, applies control, steps physics, checks constraints.

```python
class SimulationEngine:

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.geometry = create_geometry(config.habitat)
        self.dynamics = create_dynamics(config.simulation.dynamics_level)
        self.tank_system = TankSystem(config.tanks)
        self.motor = SpinMotor(config.motor)
        self.sensors = SensorSuite(config.sensors, self.geometry)
        self.scenario = Scenario(config.disturbances)
        self.monitor = ConservationMonitor()
        self.state = SimState(config)
        self.t = 0.0

        # Precompute static geometry data
        self.sector_positions = self.geometry.compute_sector_positions(config.sectors)
        self.tank_positions = self.geometry.compute_tank_positions(config.tanks)
        self.structural_inertia = self.geometry.compute_structural_inertia()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        """
        Advance simulation by one control interval (control_dt).
        Internally sub-steps at physics dt.

        Args:
            action: (36,) normalised valve commands from RL agent

        Returns:
            observation, reward, terminated, info
        """
        n_substeps = int(self.config.simulation.control_dt / self.config.simulation.dt)

        for _ in range(n_substeps):
            # Get current disturbance state (crew/cargo masses)
            sector_masses = self.scenario.get_sector_masses(self.t)

            # Get motor torque
            motor_torque = self.motor.get_torque(self.t)

            # RK4 step
            self.state.x = rk4_step(
                self.dynamics.compute_derivatives,
                self.t, self.state.x,
                sector_masses, action, motor_torque,
                self.config, self.config.simulation.dt
            )

            # Post-step: normalise quaternion, clip tanks, update manifolds
            self.state.quaternion[:] /= np.linalg.norm(self.state.quaternion)
            self.tank_system.enforce_constraints(self.state)

            # Conservation checks
            self.monitor.check(self.t, self.state, self.config)

            self.t += self.config.simulation.dt

        # Build observation and reward
        obs = self.sensors.observe(self.state, sector_masses)
        reward = self.compute_reward(self.state, action)
        terminated = self.t >= self.config.simulation.duration

        return obs, reward, terminated, self.monitor.get_info()
```

---

## 4. Data Flow

```
                    ┌─────────────────────┐
                    │  ExperimentConfig    │
                    │  (YAML/JSON)         │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                 ▼
     ┌──────────────┐  ┌────────────┐  ┌──────────────┐
     │  Geometry     │  │  Scenario   │  │  TankSystem   │
     │  (structural  │  │  (crew/cargo│  │  (36 tanks +  │
     │   inertia)    │  │   masses)   │  │   3 manifolds)│
     └──────┬───────┘  └─────┬──────┘  └──────┬───────┘
            │                │                 │
            └───────────┬────┴────┬────────────┘
                        ▼         ▼
                 ┌──────────────────────┐
                 │   DynamicsModel       │
                 │   (Level 1: rigid     │
                 │    body equations)    │
                 └──────────┬───────────┘
                            │  dx/dt
                            ▼
                 ┌──────────────────────┐
                 │   RK4 Integrator      │
                 │   (sub-steps at dt)   │
                 └──────────┬───────────┘
                            │  x(t+Δt)
                            ▼
           ┌────────────────┼────────────────┐
           ▼                ▼                 ▼
   ┌──────────────┐  ┌────────────┐  ┌──────────────┐
   │  Sensors      │  │  Monitor    │  │  Recorder    │
   │  (obs vector) │  │  (H, E, q) │  │  (→ database)│
   └──────┬───────┘  └────────────┘  └──────────────┘
          │ observation
          ▼
   ┌──────────────┐
   │  Gymnasium    │
   │  Environment  │──── obs, reward, done ───▶  SAC Agent
   │              │◀─── action (36 valves) ───  (MLP policy)
   └──────────────┘
```

---

## 5. Key Implementation Details

### 5.1 Inertia Tensor Computation (`core/inertia.py`)

This is called at every RK4 sub-step. Must be fast.

```python
def compute_inertia_tensor(
    structural_inertia: np.ndarray,      # (3,3) precomputed, constant
    sector_positions: np.ndarray,         # (36, 3) precomputed, constant
    sector_masses: np.ndarray,            # (36,) time-varying
    tank_positions: np.ndarray,           # (36, 3) precomputed, constant
    tank_masses: np.ndarray,              # (36,) time-varying (from state)
    manifold_positions: np.ndarray,       # (3, 3) precomputed, constant
    manifold_masses: np.ndarray           # (3,) time-varying (from state)
) -> np.ndarray:
    """Compute total inertia tensor about geometric centre. Returns (3,3)."""

    I = structural_inertia.copy()

    # Vectorised point-mass contributions: I += Σ m_k * (|r_k|² I₃ - r_k r_kᵀ)
    # Combine sectors and tanks into one batch for efficiency
    all_positions = np.concatenate([sector_positions, tank_positions, manifold_positions])
    all_masses = np.concatenate([sector_masses, tank_masses, manifold_masses])

    # r_squared: (N,) — |r_k|²
    r_sq = np.sum(all_positions ** 2, axis=1)

    # Outer products: (N, 3, 3) — r_k r_kᵀ
    outer = all_positions[:, :, None] * all_positions[:, None, :]

    # Weighted sum
    I += np.sum(all_masses[:, None, None] * (r_sq[:, None, None] * np.eye(3) - outer), axis=0)

    return I
```

### 5.2 Quaternion Utilities (`core/quaternion.py`)

Minimal, no-allocation implementations.

```python
def quat_multiply(p, q):
    """Hamilton product of two quaternions [w, x, y, z]."""
    ...

def quat_to_rotation_matrix(q):
    """Convert unit quaternion to 3x3 rotation matrix."""
    ...

def quat_to_euler_zxz(q):
    """Extract precession (ψ), nutation (θ), spin (φ) angles."""
    ...

def omega_matrix(w):
    """Build 4x4 Ω matrix for quaternion kinematics."""
    ...

def quat_rotate_vector(q, v):
    """Rotate vector v by quaternion q. Returns rotated vector."""
    ...
```

### 5.3 RK4 with Quaternion Normalisation (`core/integrator.py`)

```python
def rk4_step(deriv_fn, t, x, *args, dt):
    """Standard RK4 step. deriv_fn signature: f(t, x, *args) -> dx/dt"""
    k1 = deriv_fn(t, x, *args)
    k2 = deriv_fn(t + dt/2, x + dt/2 * k1, *args)
    k3 = deriv_fn(t + dt/2, x + dt/2 * k2, *args)
    k4 = deriv_fn(t + dt, x + dt * k3, *args)
    return x + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
    # Caller is responsible for quaternion normalisation after this returns
```

### 5.4 Tank System (`actuators/tank_system.py`)

Encapsulates the hybrid manifold logic, constraint enforcement, and axial equalisation.

```python
class TankSystem:

    def __init__(self, config: TankConfig):
        self.config = config
        self.n_per_station = config.n_tanks_per_station  # 12
        self.n_stations = config.n_stations              # 3

    def compute_tank_derivatives(
        self,
        tank_masses: np.ndarray,     # (36,)
        manifold_masses: np.ndarray, # (3,)
        valve_commands: np.ndarray   # (36,) in [-1, +1]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (d_tank_masses/dt, d_manifold_masses/dt).
        Handles constraint clipping and axial equalisation.
        """
        # Circumferential: valve flow per tank
        flow_rates = valve_commands * self.config.q_circ_max        # (36,)

        # Clip at tank limits (can't drain empty tank, can't overfill full tank)
        flow_rates = self._clip_at_limits(flow_rates, tank_masses)

        # Manifold drain rates: net flow per station
        flow_2d = flow_rates.reshape(self.n_stations, self.n_per_station)
        manifold_drain = -flow_2d.sum(axis=1)                       # (3,)

        # Throttle if manifold would go negative
        manifold_drain = self._throttle_manifold(manifold_drain, manifold_masses)

        # Axial equalisation: proportional controller on manifold level differences
        axial_flow = self._axial_equalisation(manifold_masses)       # (3,) net flow per manifold

        d_tanks = flow_rates
        d_manifolds = manifold_drain + axial_flow

        return d_tanks, d_manifolds

    def _axial_equalisation(self, manifold_masses):
        """
        Automatic pressure equalisation between manifolds.
        Drives manifold levels toward their mean at rate q_axial_max.
        """
        mean_level = manifold_masses.mean()
        error = mean_level - manifold_masses      # positive = under-filled
        # Proportional transfer, capped at q_axial_max
        flow = np.clip(error * K_AXIAL, -self.config.q_axial_max, self.config.q_axial_max)
        # Ensure conservation: flows must sum to zero
        flow -= flow.mean()
        return flow

    def enforce_constraints(self, state: SimState):
        """Hard-clip tanks to [0, max] and fix any conservation drift."""
        np.clip(state.tank_masses, 0.0, self.config.tank_capacity, out=state.tank_masses)
        np.clip(state.manifold_masses, 0.0, None, out=state.manifold_masses)

        # Fix total water mass conservation
        total = state.tank_masses.sum() + state.manifold_masses.sum()
        drift = total - self.config.total_water_mass
        # Distribute correction to manifolds proportionally
        state.manifold_masses -= drift * state.manifold_masses / state.manifold_masses.sum()
```

### 5.5 Gymnasium Environment (`environment/habitat_env.py`)

Standard Gymnasium interface for SAC compatibility.

```python
class HabitatEnv(gymnasium.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config
        self.engine = SimulationEngine(config)

        n_obs = 3 * config.sensors.n_accelerometers + 75  # 93 for 6 accels
        n_act = config.tanks.n_tanks_per_station * config.tanks.n_stations  # 36

        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float64
        )
        self.action_space = gymnasium.spaces.Box(
            low=-1.0, high=1.0, shape=(n_act,), dtype=np.float64
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.engine = SimulationEngine(self.config)
        obs = self.engine.sensors.observe(
            self.engine.state,
            self.engine.scenario.get_sector_masses(0.0)
        )
        return obs, {}

    def step(self, action):
        obs, reward, terminated, info = self.engine.step(action)
        truncated = False
        return obs, reward, terminated, truncated, info
```

---

## 6. Database Schema (`database/schema.py`)

Using SQLite via SQLAlchemy for portability. Three core tables.

```
┌──────────────────────┐
│  experiments          │
├──────────────────────┤
│  id (PK)             │
│  name                │
│  created_at          │
│  config_json         │  ← full ExperimentConfig serialised
│  notes               │
│  dynamics_level      │
│  status              │  ← "running", "completed", "failed"
└──────────┬───────────┘
           │ 1:N
           ▼
┌──────────────────────┐
│  episodes            │
├──────────────────────┤
│  id (PK)             │
│  experiment_id (FK)  │
│  episode_number      │
│  total_reward        │
│  duration            │
│  seed                │
│  final_nutation_deg  │  ← summary metrics for fast queries
│  max_cm_offset_m     │
│  mean_pump_energy    │
│  h_conservation_err  │
└──────────┬───────────┘
           │ 1:N
           ▼
┌──────────────────────┐
│  snapshots           │
├──────────────────────┤
│  id (PK)             │
│  episode_id (FK)     │
│  t                   │
│  state_blob          │  ← full state vector as binary (compact)
│  observation_blob    │  ← observation vector as binary
│  action_blob         │  ← action vector as binary
│  reward              │
│  h_inertial (3)      │  ← angular momentum in inertial frame
│  kinetic_energy      │
│  cm_offset (3)       │
│  nutation_angle      │
└──────────────────────┘
```

Snapshot recording is configurable: every N steps, or at control intervals only, or only at the end. For RL training (thousands of episodes), record only episode-level summaries. For analysis runs, record full time histories.

---

## 7. Extension Points for Levels 2 and 3

### 7.1 State Vector Growth

Level 2 adds modal coordinates and velocities:
```
x_L2 = [ ...Level 1 (46)..., η_1, η_2, dη_1/dt, dη_2/dt ]   (50 states)
```

Level 3:
```
x_L3 = [ ...Level 1 (46)..., η_1..η_20, dη_1/dt..dη_20/dt ]  (86 states)
```

The `SimState` class reads the state dimension from config and adjusts index maps.

### 7.2 Dynamics Model Swap

```python
def create_dynamics(level: int) -> DynamicsModel:
    if level == 1:
        return RigidBodyDynamics()
    elif level == 2:
        return Modal2Dynamics()
    elif level == 3:
        return Modal20Dynamics()
```

The integrator and engine code are unchanged. Only the `compute_derivatives` implementation differs.

### 7.3 Observation Space Growth

Level 2+ adds vibration content to accelerometer readings. The accelerometer model uses the dynamics model's output to compute a_meas including modal contributions. The observation vector shape grows only if we add modal state feedback to the agent (optional — the accelerometers already contain the modal information implicitly).

---

## 8. Testing Strategy

### 8.1 Unit Tests — Physics

| Test | Validates |
|------|-----------|
| Torque-free symmetric spinner | ω_z = const, ω_x = ω_y = 0, H conserved |
| Torque-free asymmetric spinner | Nutation at predicted frequency, H conserved |
| Spin-up to target rate | ω_z reaches target, correct time |
| Single mass imbalance | CM offset correct, conical whirl at spin rate |
| Tank transfer | Water conservation exact, manifold levels equalise |
| Quaternion norm preservation | \|q\| = 1 after long integration |

### 8.2 Integration Tests

| Test | Validates |
|------|-----------|
| Full episode with no control | Physics runs, conservation holds, data records |
| Full episode with random actions | No crashes, constraints enforced, DB populated |
| Gymnasium env check | `gymnasium.utils.env_checker.check_env(env)` passes |
| Config round-trip | Save config to JSON, reload, run identical simulation |

### 8.3 Benchmark Tests

| Test | Target |
|------|--------|
| 1-hour simulation at 100 Hz (Level 1, Python) | < 10 seconds wall clock |
| 1000 episodes (Level 1, Python) | < 3 hours |
| Config sweep: 100 geometries | Parallelisable, linear scaling |

---

## 9. Dependencies

```
# Core
numpy >= 1.24
scipy >= 1.10          # for fallback integrators if needed

# RL
gymnasium >= 0.29
stable-baselines3 >= 2.0
torch >= 2.0

# Database
sqlalchemy >= 2.0
alembic                # schema migrations

# Visualization
matplotlib >= 3.7

# Configuration
pyyaml
pydantic >= 2.0        # for config validation (alternative to raw dataclasses)

# Testing
pytest
pytest-benchmark
```

---

## 10. Implementation Order

The build sequence is designed so each step produces a testable, runnable artifact.

### Phase 1: Physics Core (can simulate and plot)
1. `config.py` — dataclasses with a reference cylinder config
2. `core/quaternion.py` — with unit tests
3. `core/inertia.py` — with unit tests against hand calculations
4. `geometry/cylinder.py` — structural inertia for the reference cylinder
5. `dynamics/rigid_body.py` — Euler equations, Level 1
6. `core/integrator.py` — RK4
7. `simulation/state.py` — state vector management
8. `actuators/motor.py` — torque profiles
9. `simulation/engine.py` — minimal engine (no tanks, no sensors, no disturbances)
10. `simulation/monitors.py` — conservation checks
11. **Milestone: torque-free spinner test passes, spin-up test passes**

### Phase 2: Disturbances and Tanks (can simulate with mass movement and control)
12. `disturbances/mass_schedule.py` — prescribed crew movement
13. `actuators/tank_system.py` — 36 tanks + hybrid manifold
14. Update `dynamics/rigid_body.py` to include tank mass in inertia computation
15. Update `simulation/engine.py` with full step logic
16. **Milestone: mass imbalance produces correct conical whirl, tanks can correct it**

### Phase 3: Sensors and Environment (can train RL agent)
17. `sensors/accelerometer.py`
18. `sensors/mass_tracker.py`
19. `environment/habitat_env.py`
20. Gymnasium env check passes
21. **Milestone: random agent runs episodes, observations and actions are correct shapes**

### Phase 4: RL Training
22. `control/sac_agent.py` — wrap SB3
23. `control/training.py` — training loop with logging
24. **Milestone: agent trains, reward improves over episodes**

### Phase 5: Database and Analysis
25. `database/schema.py`
26. `database/recorder.py`
27. `database/queries.py`
28. `visualization/plots.py`
29. **Milestone: experiments stored, time series extractable, plots generated**

### Phase 6: Extended Geometry and Scenarios
30. `geometry/ring.py`, `geometry/toroid.py`
31. `disturbances/stochastic.py`
32. `disturbances/scenario.py` — composite scenarios
33. `visualization/scene_3d.py`
