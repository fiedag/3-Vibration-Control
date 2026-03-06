"""Phase 5 tests: SQLite telemetry database.

Milestone criteria:
  1. get_engine(":memory:") creates all tables without error
  2. ExperimentRecorder inserts Experiment row on entry
  3. record_episode() inserts correct Episode + Timestep rows
  4. get_reward_curve() returns data matching what was inserted
  5. list_experiments() returns correct experiment metadata
  6. DB records survive context manager exit (no rollback on clean exit)
"""

from __future__ import annotations

import pytest

from habitat_sim.config import reference_config, MotorConfig, SimulationConfig
from habitat_sim.database.schema import Experiment, Episode, Timestep, get_engine
from habitat_sim.database.recorder import ExperimentRecorder
from sqlalchemy.orm import Session


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_steps(n: int = 5) -> list[dict]:
    return [
        {
            "step_index": i,
            "t": i * 0.1,
            "omega": [0.0, 0.0, 0.2],
            "cm_offset_mag": 0.01,
            "total_water": 1800.0,
            "kinetic_energy": 100.0,
            "reward": -1.0,
            "n_violations": 0,
        }
        for i in range(n)
    ]


def _make_cfg():
    cfg = reference_config()
    cfg.motor = MotorConfig(profile="off")
    cfg.simulation = SimulationConfig(dt=0.01, duration=10.0, control_dt=0.1)
    return cfg


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class TestSchema:

    def test_create_tables_in_memory(self):
        engine = get_engine(":memory:")
        with engine.connect() as conn:
            tables = engine.dialect.get_table_names(conn)
        for name in ("experiments", "episodes", "timesteps"):
            assert name in tables

    def test_distinct_in_memory_engines(self):
        engine1 = get_engine(":memory:")
        engine2 = get_engine(":memory:")
        assert engine1 is not engine2


# ---------------------------------------------------------------------------
# ExperimentRecorder
# ---------------------------------------------------------------------------

class TestExperimentRecorder:

    def test_creates_experiment_on_entry(self):
        cfg = _make_cfg()
        with ExperimentRecorder(":memory:", "test_run", cfg) as rec:
            with Session(rec._engine) as session:
                count = session.query(Experiment).count()
        assert count == 1

    def test_experiment_name_stored(self):
        cfg = _make_cfg()
        with ExperimentRecorder(":memory:", "my_experiment", cfg) as rec:
            with Session(rec._engine) as session:
                exp = session.query(Experiment).first()
                name = exp.name
        assert name == "my_experiment"

    def test_record_episode_inserts_rows(self):
        cfg = _make_cfg()
        steps = _make_steps(5)
        with ExperimentRecorder(":memory:", "run", cfg) as rec:
            rec.record_episode(0, steps)
            with Session(rec._engine) as session:
                ep_count = session.query(Episode).count()
                ts_count = session.query(Timestep).count()
        assert ep_count == 1
        assert ts_count == 5

    def test_episode_n_steps_correct(self):
        cfg = _make_cfg()
        steps = _make_steps(8)
        with ExperimentRecorder(":memory:", "run", cfg) as rec:
            rec.record_episode(0, steps)
            with Session(rec._engine) as session:
                ep = session.query(Episode).first()
                n = ep.n_steps
        assert n == 8

    def test_total_reward_correct(self):
        cfg = _make_cfg()
        steps = _make_steps(4)  # reward = -1.0 each
        with ExperimentRecorder(":memory:", "run", cfg) as rec:
            rec.record_episode(0, steps)
            with Session(rec._engine) as session:
                ep = session.query(Episode).first()
                r = ep.total_reward
        assert r == pytest.approx(-4.0)

    def test_multiple_episodes(self):
        cfg = _make_cfg()
        with ExperimentRecorder(":memory:", "run", cfg) as rec:
            for ep_num in range(3):
                rec.record_episode(ep_num, _make_steps(5))
            with Session(rec._engine) as session:
                ep_count = session.query(Episode).count()
        assert ep_count == 3

    def test_records_survive_context_exit(self):
        cfg = _make_cfg()
        engine_ref = None
        with ExperimentRecorder(":memory:", "run", cfg) as rec:
            rec.record_episode(0, _make_steps(3))
            engine_ref = rec._engine
        with Session(engine_ref) as session:
            count = session.query(Episode).count()
        assert count == 1


# ---------------------------------------------------------------------------
# Timestep data correctness
# ---------------------------------------------------------------------------

class TestTimestepData:

    def test_timestep_t_values(self):
        cfg = _make_cfg()
        with ExperimentRecorder(":memory:", "run", cfg) as rec:
            rec.record_episode(0, _make_steps(3))
            with Session(rec._engine) as session:
                rows = (
                    session.query(Timestep)
                    .order_by(Timestep.step_index)
                    .all()
                )
                t_vals = [r.t for r in rows]
        assert t_vals == pytest.approx([0.0, 0.1, 0.2])

    def test_omega_z_stored(self):
        cfg = _make_cfg()
        with ExperimentRecorder(":memory:", "run", cfg) as rec:
            rec.record_episode(0, _make_steps(2))
            with Session(rec._engine) as session:
                row = session.query(Timestep).first()
                oz = row.omega_z
        assert oz == pytest.approx(0.2)

    def test_engine_info_stored_on_episode(self):
        cfg = _make_cfg()
        with ExperimentRecorder(":memory:", "run", cfg) as rec:
            rec.record_episode(0, _make_steps(2), engine_info={
                "final_nutation_deg": 1.5,
                "final_cm_offset_mag": 0.03,
                "final_omega_z": 0.21,
            })
            with Session(rec._engine) as session:
                ep = session.query(Episode).first()
                nutation = ep.final_nutation_deg
        assert nutation == pytest.approx(1.5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
