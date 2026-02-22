#!/usr/bin/env python3
"""
Piper Arm Core — Math and Controller
======================================
Standalone module for the hanging Piper arm.
Contains DH parameters, FK, Jacobian, IK, and PiperHangingController.
No dependency on parent-directory files.
"""

import time
import math
import numpy as np


# ============================================================
# PIPER ARM DH PARAMETERS (Modified DH, newer firmware 0x01)
# ============================================================
# Units: meters, radians
DH_ALPHA = [0, -math.pi / 2, 0, math.pi / 2, -math.pi / 2, math.pi / 2]
DH_A     = [0, 0, 0.28503, -0.02198, 0, 0]
DH_D     = [0.123, 0, 0, 0.25075, 0, 0.091]
DH_THETA_OFFSET = [
    0,
    -math.pi * 172.22 / 180,
    -math.pi * 102.78 / 180,
    0,
    0,
    0,
]

# Joint limits (radians) from URDF
JOINT_LOWER = [-2.618, 0.0, -2.967, -1.745, -1.22, -2.0944]
JOINT_UPPER = [ 2.618, 3.14,  0.0,   1.745,  1.22,  2.0944]

# HOME = midpoint of each joint's range (hanging posture)
HOME_POSITION  = [(lo + hi) / 2.0 for lo, hi in zip(JOINT_LOWER, JOINT_UPPER)]

# RELAX = arm fully hanging down (elbow at max bend, everything else zero)
RELAX_POSITION = [0.0, math.pi / 2, JOINT_LOWER[2], 0.0, 0.0, 0.0]

# Safe waypoint — all joints zero (pass through before HOME / RELAX)
SAFE_POSITION  = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Conversion: radians <-> 0.001 degrees (piper_sdk unit)
RAD_TO_MDEG = 1000.0 * 180.0 / math.pi  # ≈ 57295.78

# Base transform for upside-down mount: 180° rotation around X.
# Flips Y and Z so that +Z = UP in the real world.
T_BASE = np.array([
    [1,  0,  0, 0],
    [0, -1,  0, 0],
    [0,  0, -1, 0],
    [0,  0,  0, 1],
], dtype=float)

# FK reference at hanging home (computed from DH + T_BASE)
FK_HOME_REFERENCE = [0.380093, 0.0, -0.427679]


# ============================================================
# FORWARD KINEMATICS
# ============================================================

def dh_transform(alpha, a, d, theta):
    """4x4 Modified DH transformation matrix."""
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array([
        [ct,      -st,       0,    a      ],
        [st * ca,  ct * ca, -sa,  -sa * d ],
        [st * sa,  ct * sa,  ca,   ca * d ],
        [0,        0,        0,    1      ],
    ])


def _raw_fk(q):
    """FK in the robot's own base frame (no base flip)."""
    transforms = [np.eye(4)]
    T = np.eye(4)
    for i in range(6):
        theta = q[i] + DH_THETA_OFFSET[i]
        T_i = dh_transform(DH_ALPHA[i], DH_A[i], DH_D[i], theta)
        T = T @ T_i
        transforms.append(T.copy())
    return transforms[6], transforms


def forward_kinematics(q):
    """
    Compute FK for the hanging Piper arm (world frame).

    Applies T_BASE (180° X flip) so that +Z = UP in the real world.

    Returns:
        T_ee: 4x4 end-effector transform (world frame)
        transforms: [T_00, T_01, ..., T_06] in world frame
    """
    _, raw_transforms = _raw_fk(q)
    transforms = [T_BASE @ T for T in raw_transforms]
    return transforms[6], transforms


def verify_fk():
    """Verify FK at hanging home against known reference."""
    T_ee, _ = forward_kinematics(HOME_POSITION)
    pos = T_ee[:3, 3]
    ref = np.array(FK_HOME_REFERENCE)
    err = np.linalg.norm(pos - ref)
    if err > 0.001:
        print(f"  WARNING: FK mismatch! Got {pos}, expected {ref}, err={err:.4f}m")
        return False
    return True


def rotation_to_euler(R):
    """Extract roll, pitch, yaw from rotation matrix (ZYX convention)."""
    if abs(R[2, 0]) < 1 - 1e-6:
        pitch = math.asin(-R[2, 0])
        roll  = math.atan2(R[2, 1] / math.cos(pitch), R[2, 2] / math.cos(pitch))
        yaw   = math.atan2(R[1, 0] / math.cos(pitch), R[0, 0] / math.cos(pitch))
    else:
        yaw = 0.0
        if R[2, 0] < 0:
            pitch = math.pi / 2
            roll  = math.atan2(R[0, 1],  R[0, 2])
        else:
            pitch = -math.pi / 2
            roll  = math.atan2(-R[0, 1], -R[0, 2])
    return roll, pitch, yaw


def euler_to_rotation(roll, pitch, yaw):
    """Create rotation matrix from roll, pitch, yaw (ZYX convention)."""
    cr, sr = math.cos(roll),  math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw),   math.sin(yaw)
    return np.array([
        [cy * cp,  cy * sp * sr - sy * cr,  cy * sp * cr + sy * sr],
        [sy * cp,  sy * sp * sr + cy * cr,  sy * sp * cr - cy * sr],
        [  -sp,          cp * sr,                  cp * cr         ],
    ])


# ============================================================
# JACOBIAN
# ============================================================

def compute_jacobian(q):
    """
    Compute 6x6 geometric Jacobian in world frame (base-flipped).

    Returns J where [v; omega] = J * dq, expressed in world frame.
    +Z = UP when robot is hanging upside-down.
    """
    _, transforms = forward_kinematics(q)
    p_ee = transforms[6][:3, 3]

    J = np.zeros((6, 6))
    for i in range(6):
        z_i = transforms[i + 1][:3, 2]
        p_i = transforms[i + 1][:3, 3]
        J[:3, i] = np.cross(z_i, p_ee - p_i)  # linear
        J[3:, i] = z_i                          # angular
    return J


# ============================================================
# INVERSE KINEMATICS
# ============================================================

def clamp_joints(q):
    """Clamp joint angles to their limits."""
    return [max(JOINT_LOWER[i], min(JOINT_UPPER[i], q[i])) for i in range(6)]


def joints_in_limits(q, margin=0.01):
    """Check if all joints are within limits."""
    return all(
        JOINT_LOWER[i] - margin <= q[i] <= JOINT_UPPER[i] + margin
        for i in range(6)
    )


def _rotation_error(R_target, R_current):
    """Compute axis-angle error vector from R_current to R_target."""
    R_err  = R_target @ R_current.T
    trace  = R_err[0, 0] + R_err[1, 1] + R_err[2, 2]
    cos_angle = max(-1.0, min(1.0, (trace - 1) / 2))
    angle  = math.acos(cos_angle)

    if angle < 1e-10:
        return np.zeros(3)

    if angle > math.pi - 1e-6:
        M     = R_err + np.eye(3)
        norms = [np.linalg.norm(M[:, i]) for i in range(3)]
        k     = int(np.argmax(norms))
        axis  = M[:, k] / norms[k]
        return angle * axis

    factor = angle / (2 * math.sin(angle))
    return factor * np.array([
        R_err[2, 1] - R_err[1, 2],
        R_err[0, 2] - R_err[2, 0],
        R_err[1, 0] - R_err[0, 1],
    ])


def ik_solve(q_init, target_pos, target_rot, max_iter=200, pos_tol=5e-4,
             rot_tol=1e-2, damping=0.05, step_limit=0.1):
    """
    Iterative damped-least-squares IK solver (full 6DOF).

    Args:
        q_init:     starting joint angles (6,)
        target_pos: desired [x, y, z] in meters
        target_rot: desired 3x3 rotation matrix
        max_iter:   maximum iterations
        pos_tol:    position convergence threshold (meters)
        rot_tol:    orientation convergence threshold (radians)
        damping:    singularity damping factor
        step_limit: max task-space step per iteration

    Returns:
        q_solution or None if failed
    """
    q          = list(q_init)
    target_pos = np.asarray(target_pos, dtype=float)

    for _ in range(max_iter):
        T_ee, _ = forward_kinematics(q)

        pos_err  = target_pos - T_ee[:3, 3]
        rot_err  = _rotation_error(target_rot, T_ee[:3, :3])
        pos_norm = np.linalg.norm(pos_err)
        rot_norm = np.linalg.norm(rot_err)

        if pos_norm < pos_tol and rot_norm < rot_tol:
            return q

        dx_vec    = np.concatenate([pos_err, rot_err])
        step_norm = np.linalg.norm(dx_vec)
        if step_norm > step_limit:
            dx_vec = dx_vec * step_limit / step_norm

        J   = compute_jacobian(q)
        JJT = J @ J.T + damping ** 2 * np.eye(6)
        dq  = J.T @ np.linalg.solve(JJT, dx_vec)

        q = [q[i] + dq[i] for i in range(6)]
        q = clamp_joints(q)

    # Accept if position is close enough even if not fully converged
    T_ee, _ = forward_kinematics(q)
    if np.linalg.norm(target_pos - T_ee[:3, 3]) < pos_tol * 5:
        return q

    return None


# ============================================================
# ROBOT INTERFACE
# ============================================================

class PiperHangingController:
    """Controller for a Piper arm mounted upside-down (hanging)."""

    def __init__(self, can_port="can0"):
        from piper_sdk import C_PiperInterface_V2
        self.piper      = C_PiperInterface_V2(can_port)
        self.speed      = 30   # percent (1-100)
        self._connected = False

    def connect(self):
        """Connect to the robot and enable motors."""
        print("  Connecting to robot ...", end=" ", flush=True)
        self.piper.ConnectPort()

        timeout = time.time() + 10.0
        while not self.piper.EnablePiper():
            if time.time() > timeout:
                print("FAILED (timeout enabling)")
                return False
            time.sleep(0.01)

        self.piper.GripperCtrl(0, 1000, 0x01, 0)
        time.sleep(0.5)

        self._connected = True
        print("Done")
        return True

    def read_joints(self):
        """Read current joint angles in radians."""
        msgs = self.piper.GetArmJointMsgs()
        js   = msgs.joint_state
        mdeg = [js.joint_1, js.joint_2, js.joint_3,
                js.joint_4, js.joint_5, js.joint_6]
        return [v / RAD_TO_MDEG for v in mdeg]

    def read_pose(self):
        """Read current EE pose from firmware.

        Returns:
            pos: [x, y, z] in meters
            rpy: [roll, pitch, yaw] in radians
        """
        msgs = self.piper.GetArmEndPoseMsgs()
        ep   = msgs.end_pose
        pos  = [ep.X_axis / 1e6, ep.Y_axis / 1e6, ep.Z_axis / 1e6]
        rpy  = [
            math.radians(ep.RX_axis / 1000.0),
            math.radians(ep.RY_axis / 1000.0),
            math.radians(ep.RZ_axis / 1000.0),
        ]
        return pos, rpy

    def send_joints(self, target_q, timeout=1.0, tolerance=0.02):
        """Send joint target and wait for convergence."""
        cmds  = [round(q * RAD_TO_MDEG) for q in target_q]
        start = time.time()
        while time.time() - start < timeout:
            self.piper.MotionCtrl_2(0x01, 0x01, self.speed, 0x00)
            self.piper.JointCtrl(*cmds)
            time.sleep(0.005)

            current = self.read_joints()
            max_err = max(abs(target_q[i] - current[i]) for i in range(6))
            if max_err < tolerance:
                return True

        return False

    def send_cartesian(self, x, y, z, roll, pitch, yaw, timeout=0.3, pos_tol=0.003):
        """Send Cartesian target via EndPoseCtrl (firmware does IK).

        Args:
            x, y, z:          target position in meters
            roll, pitch, yaw: target orientation in radians
        """
        X  = round(x     * 1e6)
        Y  = round(y     * 1e6)
        Z  = round(z     * 1e6)
        RX = round(math.degrees(roll)  * 1000)
        RY = round(math.degrees(pitch) * 1000)
        RZ = round(math.degrees(yaw)   * 1000)

        initial_pos, _ = self.read_pose()
        start = time.time()
        while time.time() - start < timeout:
            self.piper.MotionCtrl_2(0x01, 0x00, self.speed, 0x00)
            self.piper.EndPoseCtrl(X, Y, Z, RX, RY, RZ)
            time.sleep(0.01)

            cur_pos, _ = self.read_pose()
            pos_err = math.sqrt(
                (x - cur_pos[0]) ** 2 +
                (y - cur_pos[1]) ** 2 +
                (z - cur_pos[2]) ** 2
            )
            if pos_err < pos_tol:
                return True

            if time.time() - start > 0.2:
                moved = math.sqrt(
                    (cur_pos[0] - initial_pos[0]) ** 2 +
                    (cur_pos[1] - initial_pos[1]) ** 2 +
                    (cur_pos[2] - initial_pos[2]) ** 2
                )
                if moved < 0.001:
                    return False

        return False

    def move_delta(self, dx, dy, dz):
        """Move EE by position delta (meters) using firmware Cartesian control."""
        norm = math.sqrt(dx * dx + dy * dy + dz * dz)
        print(f"  Moving dx={dx*100:.1f} dy={dy*100:.1f} dz={dz*100:.1f} cm ...", end=" ", flush=True)

        cur_pos, cur_rpy = self.read_pose()
        tx = cur_pos[0] + dx
        ty = cur_pos[1] + dy
        tz = cur_pos[2] + dz
        t  = max(1.0, norm * 15)

        if self.send_cartesian(tx, ty, tz, cur_rpy[0], cur_rpy[1], cur_rpy[2], timeout=t):
            print("Done")
            return True

        # Fallback: reduce each axis by 1-3 cm
        for step in [1, 2, 3]:
            for axis in range(3):
                d = [dx, dy, dz]
                if d[axis] > 0:
                    d[axis] -= step * 0.01
                elif d[axis] < 0:
                    d[axis] += step * 0.01
                else:
                    continue
                ntx = cur_pos[0] + d[0]
                nty = cur_pos[1] + d[1]
                ntz = cur_pos[2] + d[2]
                n   = math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
                if self.send_cartesian(ntx, nty, ntz, cur_rpy[0], cur_rpy[1], cur_rpy[2],
                                       timeout=max(1.0, n * 15)):
                    print(f"Done (adjusted: {d[0]*100:.0f} {d[1]*100:.0f} {d[2]*100:.0f} cm)")
                    return True

        print("unreachable")
        return False

    def move_to(self, x, y, z, roll=None, pitch=None, yaw=None):
        """Move EE to absolute position (and optionally orientation)."""
        print(f"  Moving to x={x*100:.1f} y={y*100:.1f} z={z*100:.1f} cm ...", end=" ", flush=True)

        if roll is None:
            _, cur_rpy = self.read_pose()
            roll, pitch, yaw = cur_rpy

        if self.send_cartesian(x, y, z, roll, pitch, yaw):
            print("Done")
            return True
        else:
            print("Timeout")
            return False

    def go_safe(self):
        """Move to safe waypoint (all joints zero)."""
        self.send_joints(SAFE_POSITION, timeout=8.0)

    def go_home(self):
        """Safe waypoint -> mid-range home."""
        print("  Going home (safe -> mid-range) ...", end=" ", flush=True)
        self.go_safe()
        if self.send_joints(HOME_POSITION, timeout=8.0):
            print("Done")
        else:
            print("Timeout")

    def go_relax(self):
        """Safe -> flex elbow -> relax."""
        print("  Relaxing (safe -> elbow -> relax) ...", end=" ", flush=True)
        self.go_safe()
        elbow_first = [0.0, 0.0, JOINT_LOWER[2], 0.0, 0.0, 0.0]
        self.send_joints(elbow_first, timeout=8.0)
        if self.send_joints(RELAX_POSITION, timeout=8.0):
            print("Done")
        else:
            print("Timeout")

    def gripper_ctrl(self, opening_mm, effort=1000):
        """Set gripper opening in mm (0 = closed, ~70 = fully open)."""
        opening_um = int(max(0, min(70000, opening_mm * 1000)))
        for _ in range(10):
            self.piper.GripperCtrl(opening_um, effort, 0x01, 0)
            time.sleep(0.005)

    def shutdown(self):
        """Clean shutdown."""
        if self._connected:
            try:
                self.piper.DisablePiper()
            except Exception:
                pass
