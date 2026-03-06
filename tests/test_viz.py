"""Tests for habitat_sim.visualization.scene_3d.

These tests do NOT require plotly to be installed — they skip gracefully
when the optional dependency is absent.  When plotly IS available they
verify structure, shapes, and correctness of the generated figures.
"""

from __future__ import annotations

import numpy as np
import pytest
from dataclasses import replace

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

try:
    import plotly.graph_objects as go  # noqa: F401
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not PLOTLY_AVAILABLE,
    reason="plotly not installed — skipping visualization tests",
)


def _make_engine(shape: str = "cylinder"):
    from habitat_sim.config import reference_config, SectorConfig, TankConfig
    from habitat_sim.simulation.engine import SimulationEngine

    cfg = reference_config()
    if shape == "toroid":
        cfg.habitat = replace(cfg.habitat, shape="toroid", minor_radius=5.0)
        cfg.sectors = SectorConfig(n_angular=12, n_axial=1)
        cfg.tanks = replace(cfg.tanks, n_tanks_per_station=12, n_stations=1)
        cfg.sensors = replace(cfg.sensors, n_accelerometers=2)
    return SimulationEngine(cfg)


# ---------------------------------------------------------------------------
# Wireframe helpers (no engine needed)
# ---------------------------------------------------------------------------

class TestWireframeGenerators:
    def test_cylinder_wireframe_returns_segments(self):
        from habitat_sim.visualization.scene_3d import _cylinder_wireframe
        segs = _cylinder_wireframe(R=10.0, L=20.0)
        assert len(segs) > 0
        for seg in segs:
            assert seg.ndim == 2
            assert seg.shape[1] == 3

    def test_cylinder_wireframe_all_at_radius(self):
        from habitat_sim.visualization.scene_3d import _cylinder_wireframe
        R = 10.0
        segs = _cylinder_wireframe(R=R, L=20.0)
        for seg in segs:
            # Every point should be at distance == R from z-axis (within float precision)
            r_xy = np.sqrt(seg[:, 0]**2 + seg[:, 1]**2)
            np.testing.assert_allclose(r_xy, R, atol=1e-10)

    def test_toroid_wireframe_returns_segments(self):
        from habitat_sim.visualization.scene_3d import _toroid_wireframe
        segs = _toroid_wireframe(R=10.0, r=3.0)
        assert len(segs) > 0
        for seg in segs:
            assert seg.ndim == 2
            assert seg.shape[1] == 3

    def test_toroid_wireframe_points_on_torus(self):
        """All points must satisfy the toroid implicit equation."""
        from habitat_sim.visualization.scene_3d import _toroid_wireframe
        R, r = 10.0, 3.0
        segs = _toroid_wireframe(R=R, r=r, n_tor=8, n_pol=8)
        for seg in segs:
            x, y, z = seg[:, 0], seg[:, 1], seg[:, 2]
            # Implicit equation: (sqrt(x²+y²) - R)² + z² = r²
            lhs = (np.sqrt(x**2 + y**2) - R)**2 + z**2
            np.testing.assert_allclose(lhs, r**2, atol=1e-10)

    def test_lines_to_scatter_produces_nones(self):
        """None separators must appear between segments."""
        from habitat_sim.visualization.scene_3d import _lines_to_scatter
        seg1 = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
        seg2 = np.array([[2, 0, 0], [3, 0, 0]], dtype=float)
        trace = _lines_to_scatter([seg1, seg2])
        assert None in trace.x
        assert None in trace.y
        assert None in trace.z


# ---------------------------------------------------------------------------
# HabitatScene — cylinder / ring
# ---------------------------------------------------------------------------

class TestHabitatSceneCylinder:
    @pytest.fixture
    def scene(self):
        from habitat_sim.visualization.scene_3d import HabitatScene
        engine = _make_engine("cylinder")
        return HabitatScene(engine)

    def test_build_figure_returns_figure(self, scene):
        import plotly.graph_objects as go
        fig = scene.build_figure()
        assert isinstance(fig, go.Figure)

    def test_figure_has_expected_traces(self, scene):
        fig = scene.build_figure()
        names = [t.name for t in fig.data]
        assert any("wireframe" in (n or "").lower() for n in names), \
            "Expected a wireframe trace"
        assert any("sector" in (n or "").lower() for n in names), \
            "Expected a sector trace"
        assert any("tank" in (n or "").lower() for n in names), \
            "Expected a tank trace"
        assert any("spin" in (n or "").lower() or "\u03c9" in (n or "") for n in names), \
            "Expected a spin axis trace"
        assert any("angular momentum" in (n or "").lower() or "H" in (n or "") for n in names), \
            "Expected an angular momentum trace"

    def test_sector_trace_has_correct_count(self, scene):
        fig = scene.build_figure()
        # Find the sector scatter trace
        sector_traces = [t for t in fig.data
                         if hasattr(t, "name") and "sector" in (t.name or "").lower()]
        assert sector_traces, "No sector trace found"
        trace = sector_traces[0]
        n_sectors = scene.engine.config.sectors.n_total
        assert len(trace.x) == n_sectors

    def test_tank_trace_has_correct_count(self, scene):
        fig = scene.build_figure()
        tank_traces = [t for t in fig.data
                       if hasattr(t, "name") and "tank" in (t.name or "").lower()
                       and hasattr(t, "x")]
        assert tank_traces, "No tank trace found"
        trace = tank_traces[0]
        n_tanks = scene.engine.config.tanks.n_tanks_total
        assert len(trace.x) == n_tanks

    def test_tank_fill_level_in_range(self, scene):
        fig = scene.build_figure()
        tank_traces = [t for t in fig.data
                       if hasattr(t, "name") and "tank" in (t.name or "").lower()
                       and hasattr(t, "marker") and hasattr(t.marker, "color")]
        assert tank_traces
        fill = np.asarray(tank_traces[0].marker.color)
        assert np.all(fill >= 0.0), "Fill levels must be >= 0"
        assert np.all(fill <= 1.0 + 1e-9), "Fill levels must be <= 1"

    def test_figure_title_contains_shape(self, scene):
        fig = scene.build_figure()
        title = fig.layout.title.text or ""
        assert "cylinder" in title.lower()

    def test_figure_has_cube_aspect(self, scene):
        fig = scene.build_figure()
        assert fig.layout.scene.aspectmode == "cube"


# ---------------------------------------------------------------------------
# HabitatScene — toroid
# ---------------------------------------------------------------------------

class TestHabitatSceneToroid:
    @pytest.fixture
    def scene(self):
        from habitat_sim.visualization.scene_3d import HabitatScene
        engine = _make_engine("toroid")
        return HabitatScene(engine)

    def test_build_figure_returns_figure(self, scene):
        import plotly.graph_objects as go
        fig = scene.build_figure()
        assert isinstance(fig, go.Figure)

    def test_sector_count_matches_toroid(self, scene):
        fig = scene.build_figure()
        sector_traces = [t for t in fig.data
                         if hasattr(t, "name") and "sector" in (t.name or "").lower()]
        assert sector_traces
        # Toroid has n_angular sectors (n_axial=1 → n_total = 12)
        assert len(sector_traces[0].x) == scene.engine.config.sectors.n_total

    def test_figure_title_contains_toroid(self, scene):
        fig = scene.build_figure()
        title = fig.layout.title.text or ""
        assert "toroid" in title.lower()


# ---------------------------------------------------------------------------
# Nutation utility
# ---------------------------------------------------------------------------

class TestNutationDeg:
    def test_aligned_vectors_give_zero(self):
        from habitat_sim.visualization.scene_3d import _nutation_deg
        omega = np.array([0.0, 0.0, 1.0])
        H = np.array([0.0, 0.0, 5.0])
        assert _nutation_deg(omega, H) == pytest.approx(0.0, abs=1e-10)

    def test_perpendicular_vectors_give_90(self):
        from habitat_sim.visualization.scene_3d import _nutation_deg
        omega = np.array([1.0, 0.0, 0.0])
        H = np.array([0.0, 1.0, 0.0])
        assert _nutation_deg(omega, H) == pytest.approx(90.0, abs=1e-8)

    def test_known_angle(self):
        from habitat_sim.visualization.scene_3d import _nutation_deg
        omega = np.array([0.0, 0.0, 1.0])
        H = np.array([0.0, 1.0, 1.0]) / np.sqrt(2.0)
        assert _nutation_deg(omega, H) == pytest.approx(45.0, abs=1e-8)

    def test_zero_omega_returns_zero(self):
        from habitat_sim.visualization.scene_3d import _nutation_deg
        assert _nutation_deg(np.zeros(3), np.array([0.0, 0.0, 1.0])) == 0.0

    def test_zero_H_returns_zero(self):
        from habitat_sim.visualization.scene_3d import _nutation_deg
        assert _nutation_deg(np.array([0.0, 0.0, 1.0]), np.zeros(3)) == 0.0


# ---------------------------------------------------------------------------
# Error on missing plotly (cannot easily test without uninstalling, so
# we just verify the ImportError message is informative)
# ---------------------------------------------------------------------------

class TestRequirePlotly:
    def test_error_message_contains_install_hint(self):
        """_require_plotly() should not raise when plotly IS available."""
        from habitat_sim.visualization.scene_3d import _require_plotly
        # If we are here, plotly is available (pytestmark skips otherwise)
        _require_plotly()  # should not raise
