"""Configuration dataclasses for the habitat simulation.

All physical parameters are specified here. Serialisable to JSON/YAML
for experiment reproducibility.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict

import numpy as np


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class HabitatConfig:
    """Structural geometry of the habitat."""
    shape: str = "cylinder"            # "cylinder", "ring", "toroid"
    radius: float = 10.0               # m  (R for cylinder/ring, R_maj for toroid)
    length: float = 20.0               # m  (L for cylinder/ring, unused for toroid)
    minor_radius: float = 2.0          # m  (r for toroid, unused otherwise)
    wall_thickness: float = 0.01       # m
    wall_density: float = 2700.0       # kg/m³  (aluminium)
    end_plate_thickness: float = 0.01  # m  (cylinder only)
    end_plate_density: float = 2700.0  # kg/m³  (cylinder only)


@dataclass
class SectorConfig:
    """Discretisation of the habitat interior into sectors."""
    n_angular: int = 12
    n_axial: int = 3                   # 1 for toroid

    @property
    def n_total(self) -> int:
        return self.n_angular * self.n_axial


@dataclass
class TankConfig:
    """Rim water tank and hybrid manifold parameters."""
    n_tanks_per_station: int = 12
    n_stations: int = 3
    tank_capacity: float = 100.0       # kg per tank
    total_water_mass: float = 1800.0   # kg  (uniform: 50 kg/tank × 36)
    initial_distribution: str = "uniform"
    q_circ_max: float = 5.0            # kg/s  circumferential pump rate limit
    q_axial_max: float = 1.0           # kg/s  axial transfer rate limit
    k_axial: float = 0.1               # 1/s   axial equalisation gain

    @property
    def n_tanks_total(self) -> int:
        return self.n_tanks_per_station * self.n_stations


@dataclass
class MotorConfig:
    """Spin motor torque profile."""
    profile: str = "trapezoidal"       # "constant", "ramp", "trapezoidal", "s_curve"
    max_torque: float = 500.0          # N·m
    ramp_time: float = 60.0            # s
    hold_time: float = 300.0           # s  (trapezoidal only)
    target_spin_rate: float = 0.2094   # rad/s  ≈ 2 rpm


@dataclass
class SensorConfig:
    """Sensor placement and noise parameters."""
    strain_gauge_noise_std: float = 10.0     # N per gauge


@dataclass
class SimulationConfig:
    """Integrator and timing parameters."""
    dt: float = 0.01                   # s   physics time step  (100 Hz)
    duration: float = 3600.0           # s   total simulation time
    control_dt: float = 0.1            # s   RL decision interval  (10 Hz)
    dynamics_level: int = 1            # 1, 2, or 3

    @property
    def n_substeps(self) -> int:
        """Number of physics steps per control step."""
        return int(round(self.control_dt / self.dt))



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

# ---------------------------------------------------------------------------
# Top-level experiment config
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Complete specification for a simulation experiment."""
    habitat: HabitatConfig = field(default_factory=HabitatConfig)
    sectors: SectorConfig = field(default_factory=SectorConfig)
    tanks: TankConfig = field(default_factory=TankConfig)
    motor: MotorConfig = field(default_factory=MotorConfig)
    sensors: SensorConfig = field(default_factory=SensorConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    disturbances: list = field(default_factory=list)
    rl: RLConfig = field(default_factory=RLConfig)
    stochastic: StochasticConfig = field(default_factory=StochasticConfig)
    seed: int = 42

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Convert to plain dict (numpy arrays → lists)."""
        d = asdict(self)
        # Convert any numpy arrays that crept in
        def _convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_convert(v) for v in obj]
            return obj
        return _convert(d)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        return cls(
            habitat=HabitatConfig(**d.get("habitat", {})),
            sectors=SectorConfig(**d.get("sectors", {})),
            tanks=TankConfig(**d.get("tanks", {})),
            motor=MotorConfig(**d.get("motor", {})),
            sensors=SensorConfig(**d.get("sensors", {})),
            simulation=SimulationConfig(**d.get("simulation", {})),
            disturbances=d.get("disturbances", []),
            rl=RLConfig(**d.get("rl", {})),
            stochastic=StochasticConfig(**d.get("stochastic", {})),
            seed=d.get("seed", 42),
        )

    @classmethod
    def from_json(cls, s: str) -> "ExperimentConfig":
        return cls.from_dict(json.loads(s))


# ---------------------------------------------------------------------------
# Reference configuration (10 m radius cylinder, ~2 rpm)
# ---------------------------------------------------------------------------

def reference_config() -> ExperimentConfig:
    """Return a sensible default config for testing and development."""
    return ExperimentConfig(
        habitat=HabitatConfig(
            shape="cylinder",
            radius=10.0,
            length=20.0,
            wall_thickness=0.01,
            wall_density=2700.0,
            end_plate_thickness=0.01,
            end_plate_density=2700.0,
        ),
        motor=MotorConfig(
            profile="trapezoidal",
            max_torque=500.0,
            ramp_time=60.0,
            hold_time=300.0,
            target_spin_rate=0.2094,   # ~2 rpm → ~0.44 g at 10 m
        ),
    )
