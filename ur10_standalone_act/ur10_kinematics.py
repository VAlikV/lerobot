"""
UR10e forward kinematics + analytic geometric Jacobian (base frame).

Extracted (verified: analytic Jacobian matches finite-difference to ~6e-7) so the
process-based OSC controller and any env share one implementation. Standard-DH,
NOMINAL parameters — `check_kinematics()` in the controller compares local FK to
the robot's calibrated TCP; on our arm that was 0.00 mm, so nominal is fine. Swap
in calibrated DH deltas here later if a tool TCP makes it diverge.
"""

import numpy as np
from scipy.spatial.transform import Rotation as Rot

# Official UR10e DH (standard convention, theta = joint angle). a,d in m; alpha in rad.
UR10E_A     = np.array([0.0,    -0.6127, -0.57155, 0.0,      0.0,      0.0])
UR10E_D     = np.array([0.1807,  0.0,     0.0,      0.17415,  0.11985,  0.11655])
UR10E_ALPHA = np.array([np.pi/2, 0.0,     0.0,      np.pi/2, -np.pi/2,  0.0])


def _dh(a, alpha, d, theta):
    """Single standard-DH link transform: Rz(theta) Tz(d) Tx(a) Rx(alpha)."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,      sa,       ca,      d],
        [0.0,     0.0,      0.0,    1.0],
    ])


def fk_jacobian(q, T_tcp=None):
    """Forward kinematics + 6x6 geometric Jacobian [linear(3); angular(3)] in base frame.

    Returns (T_ee 4x4, J 6x6).
    """
    T = np.eye(4)
    axes, origins = [], []
    for i in range(6):
        axes.append(T[:3, 2].copy())
        origins.append(T[:3, 3].copy())
        T = T @ _dh(UR10E_A[i], UR10E_ALPHA[i], UR10E_D[i], q[i])

    T_ee = T @ T_tcp if T_tcp is not None else T
    p_e = T_ee[:3, 3]

    J = np.zeros((6, 6))
    for i in range(6):
        z, p = axes[i], origins[i]
        J[:3, i] = np.cross(z, p_e - p)
        J[3:, i] = z
    return T_ee, J


def pose_to_T(pose):
    """[x,y,z,rx,ry,rz] (rotation vector) -> 4x4 homogeneous transform."""
    T = np.eye(4)
    T[:3, :3] = Rot.from_rotvec(np.asarray(pose[3:])).as_matrix()
    T[:3, 3] = pose[:3]
    return T


def pose_error(p_d, R_d, p, R):
    """6-DOF error in base frame: [position diff (3); axis-angle orientation (3)]."""
    e_p = np.asarray(p_d) - np.asarray(p)
    e_o = (R_d * R.inv()).as_rotvec()
    return np.concatenate([e_p, e_o])
