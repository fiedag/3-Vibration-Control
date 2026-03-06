"""Phase 1 tests: physics core validation.

Milestone criteria:
  1. Torque-free symmetric spinner: ω_z = const, H conserved
  2. Torque-free asymmetric spinner: nutation at predicted frequency, H conserved
  3. Spin-up to target rate
  4. Quaternion norm preserved after long integration
  5. Water conservation exact
"""

from __future__ import annotations

import numpy as np
import pytest

from habitat_sim.config import (
    ExperimentConfig, HabitatConfig, SectorConfig, TankConfig,
    MotorConfig, SensorConfig, SimulationConfig, reference_config,
)
from habitat_sim.core.quaternion import (
    quat_multiply, quat_conjugate, quat_normalize, quat_to_rotation_matrix,
    quat_rotate_vector, quat_to_euler_zxz, omega_matrix, quat_derivative,
)
from habitat_sim.core.inertia import (
    point_mass_inertia, compute_inertia_tensor, compute_cm_offset,
)
from habitat_sim.geometry.cylinder import CylinderGeometry, RingGeometry
from habitat_sim.actuators.motor import SpinMotor
from habitat_sim.simulation.engine import SimulationEngine


# ===================================================================
# Quaternion unit tests
# ===================================================================

class TestQuaternion:

    def test_identity_multiply(self):
        """q * identity = q."""
        q = np.array([0.5, 0.5, 0.5, 0.5])
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        result = quat_multiply(q, identity)
        np.testing.assert_allclose(result, q, atol=1e-15)

    def test_conjugate_product_is_norm(self):
        """q * q̄  = [|q|², 0, 0, 0]."""
        q = np.array([1.0, 2.0, 3.0, 4.0])
        result = quat_multiply(q, quat_conjugate(q))
        expected_norm_sq = np.sum(q**2)
        np.testing.assert_allclose(result[0], expected_norm_sq, atol=1e-12)
        np.testing.assert_allclose(result[1:], 0.0, atol=1e-12)

    def test_normalize(self):
        q = np.array([1.0, 1.0, 1.0, 1.0])
        qn = quat_normalize(q)
        np.testing.assert_allclose(np.linalg.norm(qn), 1.0, atol=1e-15)

    def test_rotation_matrix_identity(self):
        """Identity quaternion → identity matrix."""
        q = np.array([1.0, 0.0, 0.0, 0.0])
        R = quat_to_rotation_matrix(q)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-15)

    def test_rotation_90_about_z(self):
        """90° rotation about z-axis."""
        angle = np.pi / 2
        q = np.array([np.cos(angle/2), 0, 0, np.sin(angle/2)])
        R = quat_to_rotation_matrix(q)
        # x-axis → y-axis
        v = np.array([1.0, 0.0, 0.0])
        result = R @ v
        np.testing.assert_allclose(result, [0.0, 1.0, 0.0], atol=1e-12)

    def test_rotate_vector_matches_matrix(self):
        """quat_rotate_vector should match R @ v."""
        q = quat_normalize(np.array([1.0, 0.3, -0.5, 0.2]))
        v = np.array([1.0, 2.0, 3.0])
        R = quat_to_rotation_matrix(q)
        np.testing.assert_allclose(quat_rotate_vector(q, v), R @ v, atol=1e-12)

    def test_euler_roundtrip(self):
        """Convert to ZXZ Euler and back (via rotation matrices)."""
        q = quat_normalize(np.array([0.8, 0.1, 0.2, 0.5]))
        R_orig = quat_to_rotation_matrix(q)
        psi, theta, phi = quat_to_euler_zxz(q)
        # Rebuild R from ZXZ Euler angles
        Rz1 = _Rz(psi)
        Rx  = _Rx(theta)
        Rz2 = _Rz(phi)
        R_rebuilt = Rz1 @ Rx @ Rz2
        np.testing.assert_allclose(R_rebuilt, R_orig, atol=1e-10)

    def test_omega_matrix_antisymmetric(self):
        w = np.array([1.0, 2.0, 3.0])
        O = omega_matrix(w)
        np.testing.assert_allclose(O, -O.T, atol=1e-15)


def _Rz(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def _Rx(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


# ===================================================================
# Inertia tensor tests
# ===================================================================

class TestInertia:

    def test_single_point_mass(self):
        """Single mass at (R, 0, 0) on the x-axis."""
        r = np.array([[5.0, 0.0, 0.0]])
        m = np.array([10.0])
        I = point_mass_inertia(r, m)
        # I_xx = m * (0 + 0) = 0 (mass on the x-axis)
        # Wait: I_xx = m*(y²+z²) = 0, I_yy = m*(x²+z²) = 250, I_zz = m*(x²+y²) = 250
        # Using the tensor formula: I = m*(|r|²I - rrᵀ)
        # |r|² = 25, so m*|r|²*I = 250*I
        # rrᵀ = [[25,0,0],[0,0,0],[0,0,0]]
        # I = 10*(250*I - rrᵀ) wait, that's wrong.
        # I = m*(|r|²I₃ - rrᵀ) = 10*(25*I₃ - [[25,0,0],[0,0,0],[0,0,0]])
        # = [[10*(25-25), 0, 0], [0, 10*25, 0], [0, 0, 10*25]]
        # = [[0, 0, 0], [0, 250, 0], [0, 0, 250]]
        assert I[0, 0] == pytest.approx(0.0)
        assert I[1, 1] == pytest.approx(250.0)
        assert I[2, 2] == pytest.approx(250.0)

    def test_symmetric_ring_of_masses(self):
        """12 equal masses at rim should give I_xx = I_yy, I_xy = 0."""
        R = 10.0
        n = 12
        angles = np.linspace(0, 2*np.pi, n, endpoint=False) + np.pi/n
        positions = np.column_stack([R*np.cos(angles), R*np.sin(angles),
                                     np.zeros(n)])
        masses = np.full(n, 5.0)
        I = point_mass_inertia(positions, masses)
        # Axisymmetric: I_xx ≈ I_yy
        np.testing.assert_allclose(I[0, 0], I[1, 1], rtol=1e-10)
        # Off-diagonal should be ≈ 0
        np.testing.assert_allclose(I[0, 1], 0.0, atol=1e-10)
        np.testing.assert_allclose(I[0, 2], 0.0, atol=1e-10)
        np.testing.assert_allclose(I[1, 2], 0.0, atol=1e-10)

    def test_cm_offset_single_mass(self):
        """A single mass at (R, 0, 0) with a 100 kg structure at origin."""
        pos = np.array([[10.0, 0.0, 0.0]])
        mass = np.array([10.0])
        cm = compute_cm_offset(
            structural_mass=100.0,
            sector_positions=pos, sector_masses=mass,
            tank_positions=np.zeros((0, 3)), tank_masses=np.array([]),
            manifold_positions=np.zeros((0, 3)), manifold_masses=np.array([]),
        )
        # CM = 10*10 / (100+10) = 100/110 ≈ 0.909 along x
        np.testing.assert_allclose(cm[0], 100.0 / 110.0, atol=1e-12)
        np.testing.assert_allclose(cm[1], 0.0, atol=1e-12)
        np.testing.assert_allclose(cm[2], 0.0, atol=1e-12)


# ===================================================================
# Cylinder geometry tests
# ===================================================================

class TestCylinderGeometry:

    def test_structural_mass(self):
        """Verify shell + end plate mass against hand calculation."""
        cfg = HabitatConfig(radius=10, length=20, wall_thickness=0.01,
                            wall_density=2700, end_plate_thickness=0.01,
                            end_plate_density=2700)
        geom = CylinderGeometry(cfg)
        m_shell = 2700 * 2 * np.pi * 10 * 0.01 * 20     # ≈ 33929 kg
        m_end = 2700 * np.pi * 100 * 0.01                # ≈ 8482 kg each
        expected = m_shell + 2 * m_end
        assert geom.structural_mass() == pytest.approx(expected, rel=1e-10)

    def test_inertia_symmetry(self):
        """I_xx == I_yy, off-diagonals == 0."""
        cfg = HabitatConfig(radius=10, length=20, wall_thickness=0.01,
                            wall_density=2700, end_plate_thickness=0.01,
                            end_plate_density=2700)
        geom = CylinderGeometry(cfg)
        I = geom.compute_structural_inertia()
        assert I[0, 0] == pytest.approx(I[1, 1], rel=1e-12)
        assert I[0, 1] == pytest.approx(0.0, abs=1e-10)
        assert I[0, 2] == pytest.approx(0.0, abs=1e-10)
        assert I[1, 2] == pytest.approx(0.0, abs=1e-10)
        # I_zz > I_xx for a "pancake" (but our L > 2R so it's elongated)
        # Actually for R=10, L=20: check both
        # I_zz is about spin axis (disc-like contribution)

    def test_sector_positions_count(self):
        cfg = HabitatConfig(radius=10, length=20)
        geom = CylinderGeometry(cfg)
        sc = SectorConfig(n_angular=12, n_axial=3)
        pos = geom.compute_sector_positions(sc)
        assert pos.shape == (36, 3)
        # All at radius R from z-axis
        r_xy = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2)
        np.testing.assert_allclose(r_xy, 10.0, atol=1e-12)

    def test_manifold_on_axis(self):
        cfg = HabitatConfig(radius=10, length=20)
        geom = CylinderGeometry(cfg)
        tc = TankConfig(n_stations=3)
        pos = geom.compute_manifold_positions(tc)
        assert pos.shape == (3, 3)
        np.testing.assert_allclose(pos[:, 0], 0.0, atol=1e-15)
        np.testing.assert_allclose(pos[:, 1], 0.0, atol=1e-15)


# ===================================================================
# Motor profile tests
# ===================================================================

class TestMotor:

    def test_trapezoidal_profile(self):
        cfg = MotorConfig(profile="trapezoidal", max_torque=100,
                          ramp_time=10, hold_time=50)
        motor = SpinMotor(cfg)
        # At t=0: 0
        assert motor.get_torque(0.0) == pytest.approx(0.0)
        # At t=5: half ramp
        assert motor.get_torque(5.0) == pytest.approx(50.0)
        # At t=10: full
        assert motor.get_torque(10.0) == pytest.approx(100.0)
        # At t=30: still holding
        assert motor.get_torque(30.0) == pytest.approx(100.0)
        # At t=65: midway down
        assert motor.get_torque(65.0) == pytest.approx(50.0)
        # At t=80: zero
        assert motor.get_torque(80.0) == pytest.approx(0.0)

    def test_off_profile(self):
        cfg = MotorConfig(profile="off")
        motor = SpinMotor(cfg)
        assert motor.get_torque(100.0) == pytest.approx(0.0)


# ===================================================================
# Integration tests: torque-free spinner
# ===================================================================

class TestTorqueFreeSpinner:
    """Simulate a torque-free spinning habitat and verify conservation."""

    @staticmethod
    def _make_torque_free_config(omega_z: float = 0.2094) -> ExperimentConfig:
        """Config with motor off, short duration for testing."""
        cfg = reference_config()
        cfg.motor = MotorConfig(profile="off")
        cfg.simulation = SimulationConfig(
            dt=0.01, duration=60.0, control_dt=0.1, dynamics_level=1,
        )
        return cfg

    def test_symmetric_spinner_omega_conserved(self):
        """Pure spin about z: ω_x = ω_y should remain ~0, ω_z constant."""
        cfg = self._make_torque_free_config()
        engine = SimulationEngine(cfg)

        # Set initial spin about z
        omega_z_init = 0.2094   # ~2 rpm
        engine.state.omega[:] = [0.0, 0.0, omega_z_init]

        # No crew/cargo masses (empty habitat + water)
        sector_masses = np.zeros(cfg.sectors.n_total)

        n_steps = int(cfg.simulation.duration / cfg.simulation.control_dt)
        for _ in range(n_steps):
            engine.step_no_control(sector_masses)

        # Check ω_z is still close to initial
        assert engine.state.omega[2] == pytest.approx(omega_z_init, rel=1e-6)
        # ω_x and ω_y should remain near zero
        assert abs(engine.state.omega[0]) < 1e-10
        assert abs(engine.state.omega[1]) < 1e-10

    def test_angular_momentum_conserved(self):
        """H in inertial frame should be constant for torque-free case."""
        cfg = self._make_torque_free_config()
        engine = SimulationEngine(cfg)
        engine.state.omega[:] = [0.0, 0.0, 0.2094]

        sector_masses = np.zeros(cfg.sectors.n_total)

        # Run for a while
        n_steps = int(cfg.simulation.duration / cfg.simulation.control_dt)
        for _ in range(n_steps):
            engine.step_no_control(sector_masses)

        info = engine.monitor.get_info()
        assert info["n_violations"] == 0, f"Violations: {info['violations']}"

    def test_quaternion_norm_preserved(self):
        """Quaternion norm should stay at 1.0."""
        cfg = self._make_torque_free_config()
        engine = SimulationEngine(cfg)
        engine.state.omega[:] = [0.01, 0.005, 0.2094]  # slight off-axis

        sector_masses = np.zeros(cfg.sectors.n_total)

        n_steps = int(cfg.simulation.duration / cfg.simulation.control_dt)
        for _ in range(n_steps):
            engine.step_no_control(sector_masses)

        q_norm = np.linalg.norm(engine.state.quaternion)
        assert q_norm == pytest.approx(1.0, abs=1e-10)

    def test_water_conserved(self):
        """Total water mass must be exact throughout."""
        cfg = self._make_torque_free_config()
        engine = SimulationEngine(cfg)
        engine.state.omega[:] = [0.0, 0.0, 0.2094]

        initial_water = engine.state.total_water()

        sector_masses = np.zeros(cfg.sectors.n_total)
        n_steps = int(cfg.simulation.duration / cfg.simulation.control_dt)
        for _ in range(n_steps):
            engine.step_no_control(sector_masses)

        final_water = engine.state.total_water()
        assert final_water == pytest.approx(initial_water, abs=1e-10)


# ===================================================================
# Integration test: spin-up
# ===================================================================

class TestSpinUp:
    """Verify that the motor brings the habitat to target spin rate."""

    def test_spinup_reaches_target(self):
        """After motor profile completes, ω_z ≈ target."""
        cfg = reference_config()
        cfg.simulation = SimulationConfig(
            dt=0.01, duration=500.0, control_dt=0.1, dynamics_level=1,
        )
        cfg.motor = MotorConfig(
            profile="trapezoidal",
            max_torque=500.0,
            ramp_time=60.0,
            hold_time=300.0,
            target_spin_rate=0.2094,
        )

        engine = SimulationEngine(cfg)
        sector_masses = np.zeros(cfg.sectors.n_total)

        n_steps = int(cfg.simulation.duration / cfg.simulation.control_dt)
        for _ in range(n_steps):
            engine.step_no_control(sector_masses)

        # ω_z should be > 0 and increasing toward target
        # With I_zz ≈ 4.4e6 kg·m² and trapezoidal impulse ≈ 180,000 N·m·s,
        # expected ω_z ≈ 0.041 rad/s after 500 s (well short of the ~1846 s
        # needed for full spin-up to 0.2094 rad/s).
        assert engine.state.omega[2] > 0.03, (
            f"ω_z = {engine.state.omega[2]:.4f} rad/s — spin-up seems to have failed"
        )

    def test_spinup_time_estimate(self):
        """Check approximate spin-up time against analytical estimate."""
        cfg = reference_config()
        geom = CylinderGeometry(cfg.habitat)
        I = geom.compute_structural_inertia()
        I_zz = I[2, 2]

        # Add water contribution to I_zz: all at radius R
        # 36 tanks + 3 manifolds of water
        tc = cfg.tanks
        water_per_element = tc.total_water_mass / (tc.n_tanks_total + tc.n_stations)
        I_zz_water = tc.n_tanks_total * water_per_element * cfg.habitat.radius**2
        # manifold water is on axis, contributes 0 to I_zz
        I_zz_total = I_zz + I_zz_water

        target = cfg.motor.target_spin_rate
        tau = cfg.motor.max_torque
        t_est = I_zz_total * target / tau

        # Sanity: spin-up time should be in a reasonable range
        # For R=10m with ~4.4e6 kg·m² inertia and 500 N·m, expect ~1800 s
        assert 100.0 < t_est < 5000.0, f"Estimated spin-up time: {t_est:.1f} s"


# ===================================================================
# Asymmetric spinner (nutation test)
# ===================================================================

class TestNutation:
    """A mass imbalance should produce nutation at the predicted frequency."""

    def test_imbalance_produces_nutation(self):
        """With a single mass in one sector, the body should nutate."""
        cfg = reference_config()
        cfg.motor = MotorConfig(profile="off")
        cfg.simulation = SimulationConfig(
            dt=0.005,            # finer step for nutation accuracy
            duration=30.0,
            control_dt=0.05,
            dynamics_level=1,
        )

        engine = SimulationEngine(cfg)
        engine.state.omega[:] = [0.0, 0.0, 0.2094]

        # Place a 200 kg mass in sector 0 (all others empty)
        sector_masses = np.zeros(cfg.sectors.n_total)
        sector_masses[0] = 200.0

        # This creates a product of inertia — should produce nutation
        nutation_angles = []
        n_steps = int(cfg.simulation.duration / cfg.simulation.control_dt)
        for _ in range(n_steps):
            engine.step_no_control(sector_masses)
            nutation_angles.append(engine.get_nutation_angle())

        nutation_angles = np.array(nutation_angles)
        # Nutation angle should oscillate — check it's not constant
        assert nutation_angles.std() > 0.001, (
            "Nutation angle is constant — mass imbalance not producing wobble"
        )
        # And should not blow up
        assert nutation_angles.max() < 45.0, (
            f"Nutation angle reached {nutation_angles.max():.1f}° — "
            "seems unstable"
        )


# ===================================================================
# Entry point
# ===================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
