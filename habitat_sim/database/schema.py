"""SQLAlchemy 2.0 schema for habitat-sim telemetry database.

Three tables:
    experiments  -- one row per training run
    episodes     -- one row per episode within a run
    timesteps    -- key per-step scalars (not full state vector)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    DateTime, Float, ForeignKey, Integer, String, Text, create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Experiment(Base):
    """One row per training run / experiment."""
    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    config_json: Mapped[str] = mapped_column(Text, nullable=False)
    seed: Mapped[int] = mapped_column(Integer, nullable=False)
    algorithm: Mapped[str] = mapped_column(String(64), nullable=False, default="SAC")

    episodes: Mapped[list["Episode"]] = relationship(
        "Episode", back_populates="experiment", cascade="all, delete-orphan"
    )


class Episode(Base):
    """One row per episode within an experiment."""
    __tablename__ = "episodes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("experiments.id"), nullable=False
    )
    episode_num: Mapped[int] = mapped_column(Integer, nullable=False)
    n_steps: Mapped[int] = mapped_column(Integer, nullable=False)
    total_reward: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    final_nutation_deg: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    final_cm_offset_mag: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    final_omega_z: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    h_violation_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    experiment: Mapped["Experiment"] = relationship("Experiment", back_populates="episodes")
    timesteps: Mapped[list["Timestep"]] = relationship(
        "Timestep", back_populates="episode", cascade="all, delete-orphan"
    )


class Timestep(Base):
    """Key per-step scalars for time-series analysis."""
    __tablename__ = "timesteps"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    episode_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("episodes.id"), nullable=False
    )
    step_index: Mapped[int] = mapped_column(Integer, nullable=False)
    t: Mapped[float] = mapped_column(Float, nullable=False)
    omega_x: Mapped[float] = mapped_column(Float, nullable=False)
    omega_y: Mapped[float] = mapped_column(Float, nullable=False)
    omega_z: Mapped[float] = mapped_column(Float, nullable=False)
    cm_offset_mag: Mapped[float] = mapped_column(Float, nullable=False)
    total_water: Mapped[float] = mapped_column(Float, nullable=False)
    kinetic_energy: Mapped[float] = mapped_column(Float, nullable=False)
    reward: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    n_violations: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    episode: Mapped["Episode"] = relationship("Episode", back_populates="timesteps")


def get_engine(db_path: str = "habitat.db"):
    """Create (or open) a SQLite engine and ensure all tables exist.

    Args:
        db_path: File path for the SQLite database, or ":memory:" for in-memory.

    Returns:
        SQLAlchemy Engine with tables created.
    """
    url = f"sqlite:///{db_path}" if db_path != ":memory:" else "sqlite:///:memory:"
    engine = create_engine(url, echo=False)
    Base.metadata.create_all(engine)
    return engine
