"""Query helpers for the habitat-sim telemetry database.

All functions return plain Python dicts / lists for easy use with
pandas or matplotlib without importing SQLAlchemy in consumer code.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from habitat_sim.database.schema import Episode, Experiment, Timestep, get_engine


def list_experiments(db_path: str) -> list[dict]:
    """Return summary of all experiments in the database.

    Returns:
        List of dicts with keys: id, name, created_at, seed, algorithm,
        n_episodes.
    """
    engine = get_engine(db_path)
    with Session(engine) as session:
        exps = session.query(Experiment).all()
        return [
            {
                "id": e.id,
                "name": e.name,
                "created_at": e.created_at.isoformat(),
                "seed": e.seed,
                "algorithm": e.algorithm,
                "n_episodes": len(e.episodes),
            }
            for e in exps
        ]


def get_reward_curve(db_path: str, experiment_id: int) -> dict:
    """Return reward over episodes for one experiment.

    Returns:
        {"episode_num": [...], "total_reward": [...]}
    """
    engine = get_engine(db_path)
    with Session(engine) as session:
        episodes = (
            session.query(Episode)
            .filter(Episode.experiment_id == experiment_id)
            .order_by(Episode.episode_num)
            .all()
        )
        return {
            "episode_num": [e.episode_num for e in episodes],
            "total_reward": [e.total_reward for e in episodes],
        }


def get_nutation_curve(db_path: str, experiment_id: int) -> dict:
    """Return final nutation angle over episodes.

    Returns:
        {"episode_num": [...], "final_nutation_deg": [...]}
    """
    engine = get_engine(db_path)
    with Session(engine) as session:
        episodes = (
            session.query(Episode)
            .filter(Episode.experiment_id == experiment_id)
            .order_by(Episode.episode_num)
            .all()
        )
        return {
            "episode_num": [e.episode_num for e in episodes],
            "final_nutation_deg": [e.final_nutation_deg for e in episodes],
        }


def get_conservation_summary(db_path: str, experiment_id: int) -> dict:
    """Return per-episode conservation violation counts.

    Returns:
        {"episode_num": [...], "h_violation_count": [...]}
    """
    engine = get_engine(db_path)
    with Session(engine) as session:
        episodes = (
            session.query(Episode)
            .filter(Episode.experiment_id == experiment_id)
            .order_by(Episode.episode_num)
            .all()
        )
        return {
            "episode_num": [e.episode_num for e in episodes],
            "h_violation_count": [e.h_violation_count for e in episodes],
        }


def get_timestep_series(
    db_path: str,
    episode_id: int,
    columns: list[str] | None = None,
) -> dict:
    """Return per-step time-series for one episode.

    Args:
        db_path:   Database path.
        episode_id: Primary key of the Episode row.
        columns:   Which columns to return (default: all scalar columns).

    Returns:
        Dict mapping column name to list of values, ordered by step_index.
    """
    _all_cols = [
        "step_index", "t", "omega_x", "omega_y", "omega_z",
        "cm_offset_mag", "total_water", "kinetic_energy", "reward",
        "n_violations",
    ]
    cols = columns or _all_cols
    engine = get_engine(db_path)
    with Session(engine) as session:
        rows = (
            session.query(Timestep)
            .filter(Timestep.episode_id == episode_id)
            .order_by(Timestep.step_index)
            .all()
        )
        return {col: [getattr(r, col) for r in rows] for col in cols}
