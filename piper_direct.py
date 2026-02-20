#!/usr/bin/env python3
"""
Piper Arm Direct Controller (No ROS/MoveIt)
============================================
Uses DH-parameter-based FK and Jacobian IK to control the Piper arm
directly via piper_sdk over CAN bus.

Usage:
    python3 piper_direct.py              # Interactive mode
    python3 piper_direct.py --can can0   # Specify CAN port
    python3 piper_direct.py 5 0 3        # One-shot: dx=5cm, dz=3cm
"""

import sys
import time
import math
import numpy as np
from piper_sdk import C_PiperInterface_V2

# ============================================================
# PIPER ARM DH PARAMETERS (Modified DH, newer firmware 0x01)
# ============================================================
# Units: meters, radians
DH_ALPHA = [0, -math.pi / 2, 0, math.pi / 2, -math.pi / 2, math.pi / 2]
DH_A = [0, 0, 0.28503, -0.02198, 0, 0]
DH_D = [0.123, 0, 0, 0.25075, 0, 0.091]
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
JOINT_UPPER = [2.618, 3.14, 0.0, 1.745, 1.22, 2.0944]

HOME_POSITION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Conversion: radians <-> 0.001 degrees (piper_sdk unit)
RAD_TO_MDEG = 1000.0 * 180.0 / math.pi  # ≈ 57295.78

# FK reference at zero joints (from piper_sdk, in meters)
FK_ZERO_REFERENCE = [0.056128, 0.0, 0.213266]


# ============================================================
# FORWARD KINEMATICS
# ============================================================

def dh_transform(alpha, a, d, theta):
    """4x4 Modified DH transformation matrix."""
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array([
        [ct, -st, 0, a],
        [st * ca, ct * ca, -sa, -sa * d],
        [st * sa, ct * sa, ca, ca * d],
        [0, 0, 0, 1],
    ])


def forward_kinematics(q):
    """
    Compute FK for the Piper arm.

    Args:
        q: list of 6 joint angles (radians)

    Returns:
        T_ee: 4x4 end-effector transform (base frame)
        transforms: [T_00, T_01, T_02, T_03, T_04, T_05, T_06]
    """
    transforms = [np.eye(4)]  # T_00 = identity (base frame)
    T = np.eye(4)
    for i in range(6):
        theta = q[i] + DH_THETA_OFFSET[i]
        T_i = dh_transform(DH_ALPHA[i], DH_A[i], DH_D[i], theta)
        T = T @ T_i
        transforms.append(T.copy())
    return transforms[6], transforms


def verify_fk():
    """Verify FK at zero joints against known SDK reference."""
    T_ee, _ = forward_kinematics(HOME_POSITION)
    pos = T_ee[:3, 3]
    ref = np.array(FK_ZERO_REFERENCE)
    err = np.linalg.norm(pos - ref)
    if err > 0.001:
        print(f"  WARNING: FK mismatch at home! Got {pos}, expected {ref}, err={err:.4f}m")
        return False
    return True


def rotation_to_euler(R):
    """Extract roll, pitch, yaw from rotation matrix (ZYX convention)."""
    if abs(R[2, 0]) < 1 - 1e-6:
        pitch = math.asin(-R[2, 0])
        roll = math.atan2(R[2, 1] / math.cos(pitch), R[2, 2] / math.cos(pitch))
        yaw = math.atan2(R[1, 0] / math.cos(pitch), R[0, 0] / math.cos(pitch))
    else:
        yaw = 0.0
        if R[2, 0] < 0:
            pitch = math.pi / 2
            roll = math.atan2(R[0, 1], R[0, 2])
        else:
            pitch = -math.pi / 2
            roll = math.atan2(-R[0, 1], -R[0, 2])
    return roll, pitch, yaw


def euler_to_rotation(roll, pitch, yaw):
    """Create rotation matrix from roll, pitch, yaw (ZYX convention)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


# ============================================================
# JACOBIAN
# ============================================================

def compute_jacobian(q):
    """
    Compute 6x6 geometric Jacobian.

    Returns J where [v; omega] = J * dq, expressed in base frame.

    In this Modified DH convention, joint i's axis is the z-axis of
    frame i (transforms[i+1]), not frame i-1.
    """
    _, transforms = forward_kinematics(q)
    p_ee = transforms[6][:3, 3]

    J = np.zeros((6, 6))
    for i in range(6):
        z_i = transforms[i + 1][:3, 2]  # z-axis of frame i (1-indexed)
        p_i = transforms[i + 1][:3, 3]  # origin of frame i
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
    """
    Compute axis-angle error vector from R_current to R_target (base frame).
    """
    R_err = R_target @ R_current.T
    trace = R_err[0, 0] + R_err[1, 1] + R_err[2, 2]
    cos_angle = max(-1.0, min(1.0, (trace - 1) / 2))
    angle = math.acos(cos_angle)

    if angle < 1e-10:
        return np.zeros(3)

    if angle > math.pi - 1e-6:
        # Near pi: extract axis from R_err + I
        M = R_err + np.eye(3)
        norms = [np.linalg.norm(M[:, i]) for i in range(3)]
        k = int(np.argmax(norms))
        axis = M[:, k] / norms[k]
        return angle * axis

    factor = angle / (2 * math.sin(angle))
    return factor * np.array([
        R_err[2, 1] - R_err[1, 2],
        R_err[0, 2] - R_err[2, 0],
        R_err[1, 0] - R_err[0, 1],
    ])


def ik_delta(q_current, dx, dy, dz, max_iter=100, pos_tol=5e-4, damping=0.01):
    """
    Compute new joint angles for a position delta move.

    Uses the 3x6 position-only Jacobian (top 3 rows). This avoids
    wrist singularities (q5 near 0) that plague the full 6x6 Jacobian,
    and lets the orientation drift freely — appropriate for delta moves
    where the user only specifies position change.
    """
    T_ee, _ = forward_kinematics(q_current)
    target_pos = T_ee[:3, 3] + np.array([dx, dy, dz])

    q = list(q_current)
    for _ in range(max_iter):
        T_ee, _ = forward_kinematics(q)
        pos_err = target_pos - T_ee[:3, 3]

        if np.linalg.norm(pos_err) < pos_tol:
            return q

        # Position-only Jacobian (3x6) — always rank 3 except at
        # shoulder/elbow singularities, immune to wrist singularity
        J_pos = compute_jacobian(q)[:3, :]
        JJT = J_pos @ J_pos.T + damping ** 2 * np.eye(3)
        dq = J_pos.T @ np.linalg.solve(JJT, pos_err)

        q = [q[i] + dq[i] for i in range(6)]
        q = clamp_joints(q)

    # Accept if close enough
    T_ee, _ = forward_kinematics(q)
    if np.linalg.norm(target_pos - T_ee[:3, 3]) < pos_tol * 5:
        return q

    return None


def ik_solve(q_init, target_pos, target_rot, max_iter=200, pos_tol=5e-4,
             rot_tol=1e-2, damping=0.05, step_limit=0.1):
    """
    Iterative damped-least-squares IK solver.

    Args:
        q_init: starting joint angles (6,)
        target_pos: desired [x, y, z] in meters
        target_rot: desired 3x3 rotation matrix
        max_iter: maximum iterations
        pos_tol: position convergence threshold (meters)
        rot_tol: orientation convergence threshold (radians)
        damping: singularity damping factor
        step_limit: max task-space step per iteration (meters)

    Returns:
        q_solution or None if failed
    """
    q = list(q_init)
    target_pos = np.asarray(target_pos, dtype=float)

    for _ in range(max_iter):
        T_ee, _ = forward_kinematics(q)

        pos_err = target_pos - T_ee[:3, 3]
        rot_err = _rotation_error(target_rot, T_ee[:3, :3])

        pos_norm = np.linalg.norm(pos_err)
        rot_norm = np.linalg.norm(rot_err)

        if pos_norm < pos_tol and rot_norm < rot_tol:
            return q

        dx_vec = np.concatenate([pos_err, rot_err])

        # Limit step size
        step_norm = np.linalg.norm(dx_vec)
        if step_norm > step_limit:
            dx_vec = dx_vec * step_limit / step_norm

        J = compute_jacobian(q)
        JJT = J @ J.T + damping ** 2 * np.eye(6)
        dq = J.T @ np.linalg.solve(JJT, dx_vec)

        q = [q[i] + dq[i] for i in range(6)]
        q = clamp_joints(q)

    # Check if close enough even if not fully converged
    T_ee, _ = forward_kinematics(q)
    if np.linalg.norm(target_pos - T_ee[:3, 3]) < pos_tol * 5:
        return q

    return None


# ============================================================
# ROBOT INTERFACE
# ============================================================

class PiperDirectController:
    """Direct controller for the Piper arm via piper_sdk (no ROS)."""

    def __init__(self, can_port="can0"):
        self.piper = C_PiperInterface_V2(can_port)
        self.speed = 30  # percent (1-100)
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

        # Enable gripper
        self.piper.GripperCtrl(0, 1000, 0x01, 0)
        time.sleep(0.5)

        self._connected = True
        print("Done")
        return True

    def read_joints(self):
        """Read current joint angles in radians."""
        msgs = self.piper.GetArmJointMsgs()
        js = msgs.joint_state
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
        ep = msgs.end_pose
        pos = [ep.X_axis / 1e6, ep.Y_axis / 1e6, ep.Z_axis / 1e6]
        rpy = [
            math.radians(ep.RX_axis / 1000.0),
            math.radians(ep.RY_axis / 1000.0),
            math.radians(ep.RZ_axis / 1000.0),
        ]
        return pos, rpy

    def send_joints(self, target_q, timeout=1.0, tolerance=0.02):
        """Send joint target and wait for convergence."""
        cmds = [round(q * RAD_TO_MDEG) for q in target_q]

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

        Uses MOVE P (point-to-point) mode. The firmware handles inverse
        kinematics and motion planning internally.

        Args:
            x, y, z: target position in meters
            roll, pitch, yaw: target orientation in radians
            timeout: max wait time in seconds
            pos_tol: position convergence tolerance in meters
        """
        # Convert to firmware units: 0.001mm and 0.001 degrees
        X = round(x * 1e6)
        Y = round(y * 1e6)
        Z = round(z * 1e6)
        RX = round(math.degrees(roll) * 1000)
        RY = round(math.degrees(pitch) * 1000)
        RZ = round(math.degrees(yaw) * 1000)

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

            # Early abort: if robot hasn't moved after 0.2s, target is unreachable
            if time.time() - start > 0.2:
                moved = math.sqrt(
                    (cur_pos[0] - initial_pos[0]) ** 2 +
                    (cur_pos[1] - initial_pos[1]) ** 2 +
                    (cur_pos[2] - initial_pos[2]) ** 2
                )
                if moved < 0.001:
                    return False

        return False

    def get_pose(self):
        """Get current EE pose (from firmware)."""
        pos, rpy = self.read_pose()
        q = self.read_joints()
        return pos, rpy, q

    def move_delta(self, dx, dy, dz):
        """Move EE by delta (meters) using firmware Cartesian control."""
        norm = math.sqrt(dx * dx + dy * dy + dz * dz)

        print(f"  Moving dx={dx*100:.1f} dy={dy*100:.1f} dz={dz*100:.1f} cm ...", end=" ", flush=True)

        cur_pos, cur_rpy = self.read_pose()
        tx = cur_pos[0] + dx
        ty = cur_pos[1] + dy
        tz = cur_pos[2] + dz

        # Timeout scales with distance: bigger moves need more time
        t = max(1.0, norm * 15)

        if self.send_cartesian(tx, ty, tz, cur_rpy[0], cur_rpy[1], cur_rpy[2], timeout=t):
            print("Done")
            return True

        # Full move unreachable — try reducing each axis by 1-3cm
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
                n = math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
                t_s = max(1.0, n * 15)
                if self.send_cartesian(ntx, nty, ntz, cur_rpy[0], cur_rpy[1], cur_rpy[2], timeout=t_s):
                    print(f"Done (adjusted: {d[0]*100:.0f} {d[1]*100:.0f} {d[2]*100:.0f} cm)")
                    return True

        print("unreachable")
        return False

    def move_to(self, x, y, z, roll=None, pitch=None, yaw=None):
        """Move EE to absolute position using firmware Cartesian control."""
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

    def go_home(self):
        """Move all joints to zero."""
        print("  Going home ...", end=" ", flush=True)
        if self.send_joints(HOME_POSITION, timeout=8.0):
            print("Done")
        else:
            print("Timeout")

    def gripper_ctrl(self, opening_mm, effort=1000):
        """Set gripper opening in mm (0=closed, ~70=fully open)."""
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


# ============================================================
# INTERACTIVE CLI
# ============================================================

def print_pose(ctrl):
    pos, rpy, q = ctrl.get_pose()
    print(f"  Position:    x={pos[0]*100:.2f}  y={pos[1]*100:.2f}  z={pos[2]*100:.2f}  (cm)")
    print(f"  Orientation: r={math.degrees(rpy[0]):.1f}  p={math.degrees(rpy[1]):.1f}  y={math.degrees(rpy[2]):.1f}  (deg)")


def print_joints(ctrl):
    q = ctrl.read_joints()
    degs = [math.degrees(a) for a in q]
    print(f"  Joints (deg): {' '.join(f'{d:8.2f}' for d in degs)}")
    print(f"  Joints (rad): {' '.join(f'{r:8.4f}' for r in q)}")


def interactive_mode(ctrl):
    print()
    print("=" * 55)
    print("  PIPER ARM DIRECT CONTROLLER")
    print("  (DH + Jacobian IK — no ROS/MoveIt)")
    print("=" * 55)
    print()
    print("Commands:")
    print("  dx dy dz        Move by delta (cm)")
    print("                  Example: 5 0 3")
    print("  goto x y z      Move to absolute position (cm)")
    print("                  Example: goto 25 0 30")
    print("  goto x y z r p y  With orientation (degrees)")
    print("  home            Move all joints to zero")
    print("  pose            Print current EE pose")
    print("  joints          Print current joint angles")
    print("  speed 1-100     Set speed percentage")
    print("  gripper open    Open gripper")
    print("  gripper close   Close gripper")
    print("  gripper <mm>    Set gripper opening (0-70)")
    print("  quit            Exit")
    print()
    print("=" * 55)

    while True:
        try:
            user_input = input("\n> ").strip()
            if not user_input:
                continue

            parts = user_input.split()
            cmd = parts[0].lower()

            if cmd in ("quit", "q", "exit"):
                break

            elif cmd == "home":
                ctrl.go_home()

            elif cmd == "pose":
                print_pose(ctrl)

            elif cmd == "joints":
                print_joints(ctrl)

            elif cmd == "speed":
                if len(parts) != 2:
                    print("Usage: speed 1-100")
                    continue
                s = int(float(parts[1]))
                s = max(1, min(100, s))
                ctrl.speed = s
                print(f"  Speed set to {s}%")

            elif cmd == "gripper":
                if len(parts) != 2:
                    print("Usage: gripper open|close|<mm>")
                    continue
                arg = parts[1].lower()
                if arg == "open":
                    ctrl.gripper_ctrl(70)
                    print("  Gripper opened")
                elif arg == "close":
                    ctrl.gripper_ctrl(0)
                    print("  Gripper closed")
                else:
                    mm = float(arg)
                    ctrl.gripper_ctrl(mm)
                    print(f"  Gripper set to {mm:.1f} mm")

            elif cmd == "goto":
                if len(parts) == 4:
                    x = float(parts[1]) / 100.0
                    y = float(parts[2]) / 100.0
                    z = float(parts[3]) / 100.0
                    ctrl.move_to(x, y, z)
                elif len(parts) == 7:
                    x = float(parts[1]) / 100.0
                    y = float(parts[2]) / 100.0
                    z = float(parts[3]) / 100.0
                    r = math.radians(float(parts[4]))
                    p = math.radians(float(parts[5]))
                    yw = math.radians(float(parts[6]))
                    ctrl.move_to(x, y, z, r, p, yw)
                else:
                    print("Usage: goto x y z [roll pitch yaw]")

            else:
                # Delta move (input in cm, convert to meters)
                if len(parts) == 3:
                    dx = float(parts[0]) / 100.0
                    dy = float(parts[1]) / 100.0
                    dz = float(parts[2]) / 100.0
                    ctrl.move_delta(dx, dy, dz)
                elif len(parts) == 1:
                    dx = float(parts[0]) / 100.0
                    ctrl.move_delta(dx, 0.0, 0.0)
                else:
                    print("Enter: dx dy dz  (e.g., 5 0 3)")

        except ValueError:
            print("Invalid input. Enter numbers like: 5 0 3")
        except KeyboardInterrupt:
            print()
            break
        except Exception as e:
            print(f"Error: {e}")


# ============================================================
# MAIN
# ============================================================

def main():
    can_port = "can0"

    # Parse flags
    positional = []
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--can" and i + 1 < len(sys.argv):
            can_port = sys.argv[i + 1]
            i += 2
            continue
        elif not sys.argv[i].startswith("--"):
            positional.append(sys.argv[i])
        i += 1

    # Verify FK implementation
    print("  Verifying FK ...", end=" ", flush=True)
    if verify_fk():
        print("OK")
    else:
        print("MISMATCH (proceeding anyway)")

    controller = PiperDirectController(can_port=can_port)

    if not controller.connect():
        print("  Failed to connect. Exiting.")
        sys.exit(1)

    # Home on startup
    controller.go_home()

    try:
        if len(positional) == 3:
            dx = float(positional[0]) / 100.0
            dy = float(positional[1]) / 100.0
            dz = float(positional[2]) / 100.0
            controller.move_delta(dx, dy, dz)
        else:
            interactive_mode(controller)
    finally:
        print("  Going home before exit ...", end=" ", flush=True)
        try:
            controller.send_joints(HOME_POSITION, timeout=8.0)
            print("Done")
        except Exception:
            print("Skipped")
        print("  Shutting down ...", end=" ", flush=True)
        controller.shutdown()
        print("Done")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
