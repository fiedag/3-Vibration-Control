"""Quaternion utilities for rigid body orientation.

Convention: q = [w, x, y, z] where w is the scalar part.
All functions operate on plain numpy arrays of shape (4,).
"""

from __future__ import annotations

import numpy as np


def quat_multiply(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions [w, x, y, z].

    Returns a new array; does not modify inputs.
    """
    pw, px, py, pz = p
    qw, qx, qy, qz = q
    return np.array([
        pw*qw - px*qx - py*qy - pz*qz,
        pw*qx + px*qw + py*qz - pz*qy,
        pw*qy - px*qz + py*qw + pz*qx,
        pw*qz + px*qy - py*qx + pz*qw,
    ])


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Quaternion conjugate: [w, -x, -y, -z]."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Return unit quaternion.  Normalises in-place if possible."""
    n = np.linalg.norm(q)
    if n < 1e-15:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def quat_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert unit quaternion to 3×3 rotation matrix (body-to-inertial).

    R maps vectors from body frame to inertial frame:
        v_inertial = R @ v_body
    """
    w, x, y, z = q
    # Pre-compute repeated products
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z

    return np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),     2*(xz + wy)],
        [    2*(xy + wz), 1 - 2*(xx + zz),     2*(yz - wx)],
        [    2*(xz - wy),     2*(yz + wx), 1 - 2*(xx + yy)],
    ])


def quat_rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q.  Equivalent to R(q) @ v."""
    # Using the sandwich product: q v q*
    # Faster than building the full rotation matrix for a single vector.
    qv = np.array([0.0, v[0], v[1], v[2]])
    result = quat_multiply(quat_multiply(q, qv), quat_conjugate(q))
    return result[1:4]


def quat_to_euler_zxz(q: np.ndarray) -> np.ndarray:
    """Extract ZXZ Euler angles (precession ψ, nutation θ, spin φ).

    Returns array [ψ, θ, φ] in radians.

    The ZXZ convention is natural for spinning bodies:
      - ψ (precession): rotation about inertial Z
      - θ (nutation): tilt of spin axis from inertial Z
      - φ (spin): rotation about body z

    Uses the rotation matrix to avoid singularities in the quaternion
    formulation (except at θ = 0, π which is inherent to Euler angles).
    """
    R = quat_to_rotation_matrix(q)

    # Nutation angle from R[2,2] = cos(θ)
    cos_theta = np.clip(R[2, 2], -1.0, 1.0)
    theta = np.arccos(cos_theta)

    if abs(np.sin(theta)) > 1e-10:
        psi = np.arctan2(R[0, 2], -R[1, 2])   # precession
        phi = np.arctan2(R[2, 0],  R[2, 1])    # spin
    else:
        # Gimbal lock: θ ≈ 0 or π — combine ψ and φ
        psi = np.arctan2(R[1, 0], R[0, 0])
        phi = 0.0

    return np.array([psi, theta, phi])


def omega_matrix(w: np.ndarray) -> np.ndarray:
    """Build 4×4 Ω matrix for quaternion kinematics.

    dq/dt = 0.5 * Ω(ω) @ q

    where ω = [ω_x, ω_y, ω_z] is the body-frame angular velocity.
    """
    wx, wy, wz = w
    return np.array([
        [ 0.0, -wx,  -wy,  -wz],
        [ wx,   0.0,  wz,  -wy],
        [ wy,  -wz,   0.0,  wx],
        [ wz,   wy,  -wx,   0.0],
    ])


def quat_derivative(q: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Compute dq/dt = 0.5 * Ω(ω) @ q."""
    return 0.5 * omega_matrix(omega) @ q
