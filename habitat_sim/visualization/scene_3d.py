"""Interactive 3-D visualisation of the rotating habitat.

Uses Plotly to produce a rotatable figure showing:

  - Wireframe surface of the habitat shape (cylinder, ring, or toroid)
  - Sector positions on the inner wall, coloured by occupant mass
  - Rim tank positions, coloured by fill level
  - Spin axis arrow (angular velocity ω direction)
  - Angular momentum vector H in the body frame

All vectors are expressed in the **body frame**, so the habitat structure
is always correctly oriented and the nutation is visible as the angle
between the ω and H arrows.

Usage::

    from habitat_sim.config import reference_config
    from habitat_sim.simulation.engine import SimulationEngine
    from habitat_sim.visualization.scene_3d import HabitatScene

    cfg = reference_config()
    engine = SimulationEngine(cfg)
    engine.reset()

    scene = HabitatScene(engine)
    fig = scene.build_figure()
    fig.show()          # opens browser / Jupyter inline
    fig.write_html("snapshot.html")  # save for later
"""

from __future__ import annotations

import numpy as np

from habitat_sim.simulation.engine import SimulationEngine
from habitat_sim.core.inertia import compute_inertia_tensor


def _require_plotly():
    try:
        import plotly.graph_objects  # noqa: F401
    except ImportError:
        raise ImportError(
            "plotly is required for 3D visualization.\n"
            "Install with: pip install habitat-sim[viz]"
        )


def _lines_to_scatter(segments: list[np.ndarray], **kwargs):
    """Convert a list of (N, 3) arrays into a single Scatter3d trace.

    Segments are separated by None values so Plotly draws them as
    disconnected polylines within one trace.
    """
    import plotly.graph_objects as go

    xs: list = []
    ys: list = []
    zs: list = []
    for seg in segments:
        xs.extend(seg[:, 0].tolist())
        xs.append(None)
        ys.extend(seg[:, 1].tolist())
        ys.append(None)
        zs.extend(seg[:, 2].tolist())
        zs.append(None)
    return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", **kwargs)


# ---------------------------------------------------------------------------
# Wireframe mesh generators
# ---------------------------------------------------------------------------

def _cylinder_wireframe(
    R: float, L: float, n_rings: int = 12, n_lons: int = 12
) -> list[np.ndarray]:
    """Return list of (N, 3) line segments for a cylindrical surface wireframe.

    Args:
        R: radius (m)
        L: length (m)
        n_rings: number of circumferential rings along the length
        n_lons:  number of longitudinal lines around the circumference
    """
    segments: list[np.ndarray] = []

    # Circumferential rings at evenly spaced z positions
    theta = np.linspace(0.0, 2.0 * np.pi, n_lons * 4 + 1)
    for z in np.linspace(-L / 2.0, L / 2.0, n_rings):
        seg = np.column_stack([R * np.cos(theta), R * np.sin(theta),
                               np.full_like(theta, z)])
        segments.append(seg)

    # Longitudinal lines at evenly spaced angular positions
    z_arr = np.linspace(-L / 2.0, L / 2.0, n_rings)
    for theta_i in np.linspace(0.0, 2.0 * np.pi, n_lons, endpoint=False):
        seg = np.column_stack([np.full_like(z_arr, R * np.cos(theta_i)),
                               np.full_like(z_arr, R * np.sin(theta_i)),
                               z_arr])
        segments.append(seg)

    return segments


def _toroid_wireframe(
    R: float, r: float, n_tor: int = 24, n_pol: int = 16
) -> list[np.ndarray]:
    """Return list of (N, 3) line segments for a toroidal surface wireframe.

    Args:
        R: major radius (m)
        r: minor radius (m)
        n_tor: number of toroidal (constant-v) circles
        n_pol: number of poloidal (constant-u) circles
    """
    segments: list[np.ndarray] = []

    # Toroidal circles — sweep around the ring at fixed poloidal angle v
    u = np.linspace(0.0, 2.0 * np.pi, n_tor * 4 + 1)
    for v in np.linspace(0.0, 2.0 * np.pi, n_pol, endpoint=False):
        rho = R + r * np.cos(v)
        seg = np.column_stack([rho * np.cos(u), rho * np.sin(u),
                               np.full_like(u, r * np.sin(v))])
        segments.append(seg)

    # Poloidal circles — sweep the cross-section at fixed toroidal angle u
    v = np.linspace(0.0, 2.0 * np.pi, n_pol * 4 + 1)
    for u_i in np.linspace(0.0, 2.0 * np.pi, n_tor, endpoint=False):
        rho = R + r * np.cos(v)
        seg = np.column_stack([rho * np.cos(u_i), rho * np.sin(u_i),
                               r * np.sin(v)])
        segments.append(seg)

    return segments


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class HabitatScene:
    """Build an interactive Plotly 3D figure from a SimulationEngine snapshot.

    All geometry is shown in the **body frame**.  The angular velocity ω and
    angular momentum H are both drawn as arrows originating from the origin,
    so the nutation angle is directly visible as the angle between them.

    Args:
        engine: A ``SimulationEngine`` instance (may be mid-simulation).
    """

    def __init__(self, engine: SimulationEngine):
        self.engine = engine

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_figure(
        self,
        sector_masses: np.ndarray | None = None,
    ):
        """Create and return a Plotly Figure for the current engine state.

        Args:
            sector_masses: Optional override for sector masses.  If ``None``
                the engine's current scenario is queried.

        Returns:
            A ``plotly.graph_objects.Figure`` ready to be shown or exported.
        """
        _require_plotly()
        import plotly.graph_objects as go

        engine = self.engine
        cfg = engine.config

        if sector_masses is None:
            sector_masses = engine.scenario.get_sector_masses(engine.t)

        tank_masses = engine.state.tank_masses
        manifold_masses = engine.state.manifold_masses
        omega = engine.state.omega

        # Compute full inertia tensor (structural + all masses)
        I = compute_inertia_tensor(
            engine.structural_inertia,
            engine.sector_positions, sector_masses,
            engine.tank_positions, tank_masses,
            engine.manifold_positions, manifold_masses,
        )
        H = I @ omega  # angular momentum in body frame

        fig = go.Figure()

        # 1. Wireframe surface
        self._add_wireframe(fig, cfg.habitat)

        # 2. Sector markers (body-frame positions; colour = occupant mass)
        self._add_sector_markers(fig, engine.sector_positions, sector_masses)

        # 3. Rim tank markers (colour = fill fraction)
        fill_level = tank_masses / cfg.tanks.tank_capacity
        self._add_tank_markers(fig, engine.tank_positions, fill_level)

        # 4. Spin axis — direction of angular velocity ω
        omega_mag = np.linalg.norm(omega)
        arrow_len = cfg.habitat.radius * 1.5
        if omega_mag > 1e-10:
            omega_dir = omega / omega_mag
            self._add_arrow(fig, np.zeros(3), omega_dir * arrow_len,
                            name="ω (spin axis)", color="royalblue")
        else:
            # Habitat at rest: draw body z-axis as spin axis placeholder
            self._add_arrow(fig, np.zeros(3), np.array([0.0, 0.0, arrow_len]),
                            name="ω (spin axis — at rest)", color="royalblue")

        # 5. Angular momentum H
        H_mag = np.linalg.norm(H)
        if H_mag > 1e-10:
            H_dir = H / H_mag
            self._add_arrow(fig, np.zeros(3), H_dir * arrow_len,
                            name="H (angular momentum)", color="crimson")

        # Layout
        R = cfg.habitat.radius
        L = cfg.habitat.length if cfg.habitat.shape != "toroid" else cfg.habitat.minor_radius * 2.0
        half_extent = max(R, L / 2.0) * 1.3
        omega_str = f"ω = {omega_mag:.4f} rad/s" if omega_mag > 1e-10 else "ω ≈ 0"
        nutation_deg = _nutation_deg(omega, H)
        title = (
            f"Habitat ({cfg.habitat.shape}) — t = {engine.t:.1f} s"
            f" | {omega_str}"
            f" | nutation = {nutation_deg:.2f}°"
        )
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis=dict(range=[-half_extent, half_extent], title="X (m)"),
                yaxis=dict(range=[-half_extent, half_extent], title="Y (m)"),
                zaxis=dict(range=[-half_extent, half_extent], title="Z (m)"),
                aspectmode="cube",
            ),
            margin=dict(l=0, r=0, b=0, t=50),
            legend=dict(x=0.01, y=0.99),
        )

        return fig

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _add_wireframe(self, fig, hab_cfg) -> None:
        """Add the habitat surface wireframe."""
        import plotly.graph_objects as go

        shape = hab_cfg.shape
        R = hab_cfg.radius

        if shape == "toroid":
            segments = _toroid_wireframe(R, hab_cfg.minor_radius)
        else:
            segments = _cylinder_wireframe(R, hab_cfg.length)

        fig.add_trace(_lines_to_scatter(
            segments,
            name="Surface wireframe",
            showlegend=True,
            line=dict(color="rgba(30, 80, 140, 0.9)", width=1),
            hoverinfo="skip",
        ))

    def _add_sector_markers(
        self, fig, positions: np.ndarray, masses: np.ndarray
    ) -> None:
        """Add sector centroid scatter markers, colour = occupant mass."""
        import plotly.graph_objects as go

        fig.add_trace(go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode="markers",
            name="Sectors",
            marker=dict(
                size=6,
                color=masses,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(
                    title=dict(text="Sector mass (kg)", side="right"),
                    x=1.02,
                    len=0.45,
                    yanchor="top",
                    y=0.95,
                ),
                opacity=0.85,
                line=dict(width=0),
            ),
            text=[f"Sector {i}<br>Mass: {m:.1f} kg"
                  for i, m in enumerate(masses)],
            hoverinfo="text",
        ))

    def _add_tank_markers(
        self, fig, positions: np.ndarray, fill_level: np.ndarray
    ) -> None:
        """Add rim tank scatter markers, colour = fill fraction [0, 1]."""
        import plotly.graph_objects as go

        fig.add_trace(go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode="markers",
            name="Rim tanks",
            marker=dict(
                size=9,
                symbol="square",
                color=fill_level,
                colorscale="RdYlBu_r",
                cmin=0.0,
                cmax=1.0,
                showscale=True,
                colorbar=dict(
                    title=dict(text="Fill level", side="right"),
                    x=1.02,
                    len=0.45,
                    yanchor="bottom",
                    y=0.05,
                    tickformat=".0%",
                ),
                opacity=0.95,
                line=dict(width=1, color="white"),
            ),
            text=[f"Tank {i}<br>Fill: {f:.1%}"
                  for i, f in enumerate(fill_level)],
            hoverinfo="text",
        ))

    def _add_arrow(
        self,
        fig,
        origin: np.ndarray,
        direction: np.ndarray,
        name: str,
        color: str,
    ) -> None:
        """Add a vector arrow: line shaft + cone arrowhead."""
        import plotly.graph_objects as go

        tip = origin + direction
        d_len = np.linalg.norm(direction)

        # Shaft
        fig.add_trace(go.Scatter3d(
            x=[origin[0], tip[0]],
            y=[origin[1], tip[1]],
            z=[origin[2], tip[2]],
            mode="lines",
            name=name,
            line=dict(color=color, width=5),
            hoverinfo="skip",
            showlegend=True,
        ))

        # Arrowhead cone
        if d_len > 1e-12:
            unit = direction / d_len
            cone_size = d_len * 0.10
            fig.add_trace(go.Cone(
                x=[tip[0]], y=[tip[1]], z=[tip[2]],
                u=[unit[0]], v=[unit[1]], w=[unit[2]],
                sizemode="absolute",
                sizeref=cone_size,
                colorscale=[[0, color], [1, color]],
                showscale=False,
                showlegend=False,
                hoverinfo="skip",
            ))


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _nutation_deg(omega: np.ndarray, H: np.ndarray) -> float:
    """Angle between ω and H in degrees (= nutation when both non-zero)."""
    omega_mag = np.linalg.norm(omega)
    H_mag = np.linalg.norm(H)
    if omega_mag < 1e-10 or H_mag < 1e-10:
        return 0.0
    cos_theta = np.clip(
        np.dot(omega / omega_mag, H / H_mag), -1.0, 1.0
    )
    return float(np.degrees(np.arccos(cos_theta)))
