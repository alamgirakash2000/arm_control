#!/usr/bin/env python3
"""
Remote Teleop 6DOF — Piper Hanging Arms (single or dual)
=========================================================
Full 6-DOF control: position (X, Y, Z) AND orientation (roll, pitch, yaw).

Supports single-arm mode:
    python3 remote_teleop_6dof.py --with_robot --left can0          # left arm only
    python3 remote_teleop_6dof.py --with_robot --right can1         # right arm only
    python3 remote_teleop_6dof.py --with_robot --left can0 --right can1  # both

Coordinate frame mapping (OpenVR → Robot hanging upside-down):
    Robot X = −VR Z   (VR forward → robot forward)
    Robot Y = +VR X   (VR right  → robot right)
    Robot Z = −VR Y   (VR up     → robot down, because hanging)

Robot command flow:
    1. Robot homes → read actual physical pose → robot_home (pos + rpy + R_mat)
    2. VR snaps   → controller pose captured  → vr_home   (pos + 3×3 rot matrix)
    3. Every frame:
       Δpos   = (vr_pos − vr_home_pos) × scale, remapped to robot frame
       ΔR_vr  = R_vr_current @ R_vr_home⁻¹
       ΔR_rob = P @ ΔR_vr @ Pᵀ              (frame transform)
       Δrpy   = euler(ΔR_rob)                (near-identity → stable for display)
       R_cmd  = euler_to_rot(Δrpy) @ R_home  (proper composition for robot)
       cmd_rpy = euler(R_cmd)
    4. send_cartesian(target_pos, cmd_rpy) → absolute 6-DOF
"""

import argparse
import math
import os
import sys
import time
import threading

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from piper_core import (
    rotation_to_euler,
    euler_to_rotation,
    forward_kinematics,
    HOME_POSITION,
    JOINT_LOWER,
    JOINT_UPPER,
    ik_solve,
    T_BASE,
    RAD_TO_MDEG,
)

from relay_server import pose_store, start_server

try:
    import cv2
    _cv2_ok = True
except ImportError:
    _cv2_ok = False


# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_SCALE      = 1.0
DEFAULT_ROT_SCALE  = 1.0
DEFAULT_SPEED      = 50
DEFAULT_SMOOTHING  = 0.5
GRIPPER_MAX_MM     = 40.0
GRIPPER_DEADBAND   = 2.0      # %
DEADBAND_CM        = 0.0      # cm — per-axis EMA deadband
MOVE_THRESH_CM     = 0.0      # cm — per-axis position threshold (0 = always update)
ROT_THRESH_DEG     = 0.0      # deg — rotation threshold (0 = always update)
UPDATE_INTERVAL    = 0.025    # seconds between accepting new targets (40 Hz)
HOLD_HZ            = 30       # Hz — rate to resend current target
DISPLAY_HZ         = 10        # terminal print rate
REF_SECS           = 2.5      # seconds to average for home pose capture
STALE_MS           = 500.0    # pose data older than this = tracking lost
MAX_POS_DELTA_CM   = 100.0    # generous backstop (joint limits are the real safety)
MAX_ROT_DELTA_DEG  = 180.0    # generous backstop (joint limits are the real safety)
JOINT_LIMIT_MARGIN = math.radians(2.0)  # 2° fixed margin from each joint limit

# IK joint-space smoothing: per-joint EMA rate (0=frozen, 1=instant)
# J4 (forearm twist) tracks slower to prevent slamming into limits
IK_JOINT_ALPHA   = [0.35, 0.35, 0.35, 0.2, 0.35, 0.35]
# IK joint weights: higher = joint moves less (penalizes J4 2x)
IK_JOINT_WEIGHTS = [1.0, 1.0, 1.0, 2.0, 1.0, 1.0]

# ── VR-to-Robot frame transform ──────────────────────────────────────────────
# OpenVR: X=right, Y=up, −Z=forward
# Robot (hanging): X=forward, Y=right, Z=down
#
#   Robot X = −VR Z    Robot Y = +VR X    Robot Z = −VR Y
#
# P maps a vector from VR frame to robot frame.
# For rotations:  R_robot = P @ R_vr @ Pᵀ

VR_TO_ROBOT = np.array([
    [ 0,  0, -1],
    [ 1,  0,  0],
    [ 0, -1,  0],
], dtype=np.float64)

# Position axis remap (same mapping, used for position deltas)
AXIS_MAP = [(1, 1), (0, 1), (2, -1)]

# Rotation part of upside-down base transform (native ↔ FK world frame)
T_BASE_ROT = np.diag([1.0, -1.0, -1.0])

# ── ANSI helpers ──────────────────────────────────────────────────────────────
GREEN  = "\033[32m"
RED    = "\033[31m"
YELLOW = "\033[33m"
RESET  = "\033[0m"

# ── cv2 overlay ──────────────────────────────────────────────────────────────
OW, OH = 1280, 720
_disp = {
    "phase": "starting", "countdown": 0, "frame": 0,
    "scale": DEFAULT_SCALE, "rot_scale": DEFAULT_ROT_SCALE,
    "sim": True, "server": "", "age_ms": -1, "arms": "both",
    "left":  dict(tracking=False, dx=0, dy=0, dz=0,
                  droll=0, dpitch=0, dyaw=0, gripper=100),
    "right": dict(tracking=False, dx=0, dy=0, dz=0,
                  droll=0, dpitch=0, dyaw=0, gripper=100),
}


# ── Math helpers ──────────────────────────────────────────────────────────────

def rotation_delta(R_cur, R_home):
    """Relative rotation: how much R_cur has rotated away from R_home."""
    return R_cur @ R_home.T


def wrap_angle(a):
    """Wrap angle to [−π, π]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def angle_ema(new, prev, alpha):
    """EMA for angles with wrapping."""
    diff = wrap_angle(new - prev)
    return wrap_angle(prev + alpha * diff)


def deadband(v, threshold):
    return 0.0 if abs(v) < threshold else v


def normalize_delta_euler(roll, pitch, yaw):
    """Normalize delta Euler angles to the representation closest to (0,0,0).

    In ZYX convention, (±π, θ, ±π) represents the same rotation as
    (0, sign·π − θ, 0).  When |roll| and |yaw| are both > 90° we are in the
    "flipped" form — switch to the compact one so that EMA smoothing and
    display stay well-behaved even when the VR controller rotates past 90°
    in the pitch direction.
    """
    if abs(roll) > math.pi / 2 and abs(yaw) > math.pi / 2:
        alt_roll  = wrap_angle(roll  - math.copysign(math.pi, roll))
        alt_pitch = wrap_angle(math.copysign(math.pi, pitch) - pitch)
        alt_yaw   = wrap_angle(yaw   - math.copysign(math.pi, yaw))
        if abs(alt_roll) + abs(alt_pitch) + abs(alt_yaw) < abs(roll) + abs(pitch) + abs(yaw):
            return alt_roll, alt_pitch, alt_yaw
    return roll, pitch, yaw


def remap(pos_delta_m, axis_map):
    return np.array([pos_delta_m[ax] * sign for ax, sign in axis_map])


def check_joint_limits(ctrl, margin=JOINT_LIMIT_MARGIN):
    """Read robot joints and check if any are near limits.

    Args:
        margin: fixed angular margin in radians (same for every joint)

    Returns (near_limit, near_details) where near_details is a list of
    (joint_number_1based, 'lo'|'hi') tuples.
    """
    try:
        joints = ctrl.read_joints()
    except Exception:
        return False, []
    near = []
    for i in range(6):
        if joints[i] <= JOINT_LOWER[i] + margin:
            near.append((i + 1, 'lo'))
        elif joints[i] >= JOINT_UPPER[i] - margin:
            near.append((i + 1, 'hi'))
    return len(near) > 0, near


def _clamp_toward_home(new_val, safe_val):
    """When near joint limit, only allow delta to move toward home or cross zero.

    Blocks delta from increasing in magnitude in the same direction.
    Allows shrinking (moving back toward home) or crossing zero.
    """
    if new_val * safe_val > 0 and abs(new_val) > abs(safe_val):
        return safe_val
    return new_val


def rotation_ema(R_new, R_prev, alpha):
    """EMA smoothing for rotation matrices via weighted average + SVD."""
    R_blend = alpha * R_new + (1.0 - alpha) * R_prev
    U, _, Vt = np.linalg.svd(R_blend)
    R_out = U @ Vt
    if np.linalg.det(R_out) < 0:
        U[:, -1] *= -1
        R_out = U @ Vt
    return R_out


def angular_distance(R1, R2):
    """Angle (radians) between two rotation matrices."""
    R_rel = R1 @ R2.T
    trace = np.clip(R_rel.trace(), -1.0, 3.0)
    return math.acos(max(-1.0, min(1.0, (trace - 1) / 2)))


def scale_rotation(R, s):
    """Scale a rotation by factor s using axis-angle representation."""
    if abs(s - 1.0) < 1e-9:
        return R
    trace = np.clip(R.trace(), -1.0, 3.0)
    angle = math.acos(max(-1.0, min(1.0, (trace - 1) / 2)))
    if angle < 1e-10:
        return np.eye(3, dtype=np.float64)
    sin_a = math.sin(angle)
    if abs(sin_a) < 1e-10:
        return R
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * sin_a)
    new_angle = angle * s
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(new_angle) * K + (1 - math.cos(new_angle)) * (K @ K)


def native_to_world_pose(pos, R):
    """Convert position + rotation from firmware native frame to FK world frame.

    T_BASE_ROT = diag(1,-1,-1) is a 180° X rotation — its own inverse.
    """
    pos_w = T_BASE_ROT @ np.asarray(pos, dtype=np.float64)
    R_w = T_BASE_ROT @ R @ T_BASE_ROT
    return list(pos_w), R_w


# ── Non-blocking robot commander ─────────────────────────────────────────────

class ArmCommander:
    def __init__(self, ctrl, label="arm"):
        self._piper   = ctrl.piper
        self._speed   = ctrl.speed
        self._lock    = threading.Lock()
        self._target  = None          # (x, y, z, roll, pitch, yaw) or None
        self._joint_target = None     # [q1..q6] or None
        self._grip_mm = None
        self._running = True
        self._label   = label
        threading.Thread(target=self._loop, daemon=True, name=f"cmd-{label}").start()

    def send(self, x, y, z, roll, pitch, yaw):
        """Update the Cartesian target pose (firmware IK mode)."""
        with self._lock:
            self._target = (x, y, z, roll, pitch, yaw)
            self._joint_target = None

    def send_joints(self, q):
        """Update the joint target (local IK mode)."""
        with self._lock:
            self._joint_target = [float(v) for v in q]
            self._target = None

    def gripper(self, mm: float):
        with self._lock:
            self._grip_mm = float(mm)

    def stop(self):
        self._running = False

    def _loop(self):
        interval = 1.0 / HOLD_HZ
        while self._running:
            with self._lock:
                tgt = self._target
                jtgt = self._joint_target
                grip = self._grip_mm
            if jtgt is not None:
                try:
                    jcmds = [round(q * RAD_TO_MDEG) for q in jtgt]
                    self._piper.MotionCtrl_2(0x01, 0x01, self._speed, 0x00)
                    self._piper.JointCtrl(*jcmds)
                except Exception:
                    pass
            elif tgt is not None:
                try:
                    x, y, z, roll, pitch, yaw = tgt
                    self._piper.MotionCtrl_2(0x01, 0x00, self._speed, 0x00)
                    self._piper.EndPoseCtrl(
                        round(x * 1e6), round(y * 1e6), round(z * 1e6),
                        round(math.degrees(roll)  * 1000),
                        round(math.degrees(pitch) * 1000),
                        round(math.degrees(yaw)   * 1000),
                    )
                except Exception:
                    pass
            if grip is not None:
                try:
                    self._piper.GripperCtrl(round(grip * 1000), 1000, 0x01, 0)
                except Exception:
                    pass
            time.sleep(interval)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Remote teleop 6DOF receiver for Piper hanging arms.",
    )
    p.add_argument("--with_robot",     action="store_true")
    p.add_argument("--left",           default=None, metavar="IFACE",
                   help="CAN interface for left arm (e.g. can0)")
    p.add_argument("--right",          default=None, metavar="IFACE",
                   help="CAN interface for right arm (e.g. can1)")
    p.add_argument("--scale",          default=DEFAULT_SCALE, type=float,
                   help="Position scale (default: 1.0)")
    p.add_argument("--rot-scale",      default=DEFAULT_ROT_SCALE, type=float,
                   help="Rotation scale (default: 1.0, >1 amplifies)")
    p.add_argument("--speed",          default=DEFAULT_SPEED, type=int)
    p.add_argument("--smoothing",      default=DEFAULT_SMOOTHING, type=float)
    p.add_argument("--ref-secs",       default=REF_SECS, type=float)
    p.add_argument("--joint-ik",      action="store_true", default=True,
                   help="Use local IK + joint control (default: on)")
    p.add_argument("--no-joint-ik",   action="store_true",
                   help="Disable local IK, use firmware Cartesian control")
    p.add_argument("--invert-gripper", action="store_true")
    p.add_argument("--port",           default=8765, type=int)
    p.add_argument("--host",           default="0.0.0.0")
    return p.parse_args()


def get_can_ports():
    env_file = "/tmp/piper_arms.env"
    if not os.path.exists(env_file):
        return None, None
    ports = {}
    try:
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    ports[k.strip()] = v.strip()
    except OSError:
        return None, None
    return ports.get("LEFT"), ports.get("RIGHT")


# ── Pose data reader ─────────────────────────────────────────────────────────

def read_latest_pose():
    data, updated_at, _ = pose_store.get()
    if data is None:
        return None
    data["_age_ms"] = (time.time() - updated_at) * 1000.0
    return data


def extract_controller(frame, hand):
    """Returns (pos, R_3x3, trigger, status_str)."""
    if frame is None:
        return None, None, 0.0, "NO DATA"
    c = frame.get(hand)
    if c is None:
        return None, None, 0.0, "NOT FOUND"
    status = c.get("status", "UNKNOWN")
    if status != "OK":
        return None, None, c.get("trigger", 0.0), status
    age = frame.get("_age_ms", 0)
    if age > STALE_MS:
        return None, None, c.get("trigger", 0.0), f"STALE ({age:.0f}ms)"
    pos = np.array(c["pos"], dtype=np.float64)
    R   = np.array(c["rot"], dtype=np.float64)
    return pos, R, float(c.get("trigger", 0.0)), "OK"


# ── Reference pose capture ────────────────────────────────────────────────────

def capture_reference(secs, poll_interval, hands):
    """Capture VR home reference for the specified hands (list of 'left'/'right')."""
    pos_samples = {h: [] for h in hands}
    R_samples   = {h: [] for h in hands}
    active_t    = 0.0
    t_wall0     = time.time()

    while active_t < secs:
        if time.time() - t_wall0 > 120.0:
            raise RuntimeError("Controllers not tracked for 2 min during reference capture.")
        frame = read_latest_pose()
        ok = {}
        for hand in hands:
            pos, R, _, status = extract_controller(frame, hand)
            ok[hand] = status == "OK"
            if ok[hand]:
                pos_samples[hand].append(pos)
                R_samples[hand].append(R)
        if all(ok.get(h, False) for h in hands):
            active_t += poll_interval
        rem = max(0.0, secs - active_t)
        status_parts = [f"{h[0].upper()}:{'OK' if ok.get(h) else 'wait'}" for h in hands]
        print(f"\r  ref remaining: {rem:4.1f}s  [{' '.join(status_parts)}]   ",
              end="", flush=True)
        if _cv2_ok:
            cv2.imshow("Remote Teleop 6DOF", render_frame())
            cv2.waitKey(1)
        time.sleep(max(0.0, poll_interval))

    print()
    out = {}
    for hand in hands:
        n = len(pos_samples[hand])
        if n < 5:
            raise RuntimeError(f"Too few reference samples for {hand.upper()} (n={n}).")
        pos_mean = np.mean(np.stack(pos_samples[hand]), axis=0)
        R_sum = np.sum(np.stack(R_samples[hand]), axis=0)
        U, _, Vt = np.linalg.svd(R_sum)
        R_mean = U @ Vt
        if np.linalg.det(R_mean) < 0:
            U[:, -1] *= -1
            R_mean = U @ Vt
        out[hand] = (pos_mean, R_mean)
    return out


# ── cv2 overlay ───────────────────────────────────────────────────────────────

def _txt(img, text, x, y, scale, color, thick=2):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thick, cv2.LINE_AA)

def _bar(img, x, y, w, h, pct, color):
    cv2.rectangle(img, (x, y), (x + w, y + h), (60, 60, 60), -1)
    fill = int(w * max(0.0, min(100.0, pct)) / 100)
    if fill > 0:
        cv2.rectangle(img, (x, y), (x + fill, y + h), color, -1)

def render_frame():
    img = np.full((OH, OW, 3), 18, dtype=np.uint8)
    d = _disp
    phase = d["phase"]

    if phase == "starting":
        _txt(img, "Starting relay server...", 250, 320, 1.8, (200, 200, 200), 3)
        _txt(img, f"Listening on {d['server']}", 200, 400, 1.2, (150, 150, 150), 2)
    elif phase == "waiting":
        _txt(img, "Waiting for VR sender data...", 80, 320, 1.6, (200, 200, 0), 3)
        _txt(img, f"Server: {d['server']}", 200, 400, 1.0, (150, 150, 150), 2)
    elif phase == "calibrating":
        _txt(img, "HOLD HOME POSITION", 160, 300, 2.2, (0, 220, 220), 4)
        _txt(img, f"Snapping home in  {d['countdown']} s", 300, 420, 1.6, (255, 255, 255), 2)
    else:
        mode_col = (0, 180, 0) if not d["sim"] else (180, 140, 0)
        mode_txt = "ROBOT" if not d["sim"] else "SIMULATION"
        age_ms = d["age_ms"]
        arm_txt = d["arms"].upper()
        _txt(img,
             f"6DOF TELEOP [{mode_txt}] [{arm_txt}]  frame {d['frame']}  "
             f"pos×{d['scale']:.1f}  rot×{d['rot_scale']:.1f}  age {age_ms:.0f}ms",
             20, 42, 0.65, mode_col, 2)
        cv2.line(img, (0, 55), (OW, 55), (60, 60, 60), 1)

        for i, side in enumerate(("left", "right")):
            s     = d[side]
            label = "LEFT" if side == "left" else "RIGHT"
            bx    = 20 + i * 640
            by    = 80

            trk_col = (0, 200, 0) if s["tracking"] else (0, 0, 200)
            trk_txt = "tracking"  if s["tracking"] else "LOST"

            _txt(img, label,   bx,       by + 36, 1.6, (230, 230, 230), 3)
            _txt(img, trk_txt, bx + 140, by + 36, 1.0, trk_col, 2)

            by += 60
            _txt(img, f"dx {s['dx']:+6.2f}cm", bx,       by, 0.8, (200, 200, 200), 2)
            _txt(img, f"dy {s['dy']:+6.2f}cm", bx + 180, by, 0.8, (200, 200, 200), 2)
            _txt(img, f"dz {s['dz']:+6.2f}cm", bx + 360, by, 0.8, (200, 200, 200), 2)

            by += 36
            _txt(img, f"R {s['droll']:+6.1f}", bx,       by, 0.8, (180, 220, 180), 2)
            _txt(img, f"P {s['dpitch']:+6.1f}", bx + 150, by, 0.8, (180, 220, 180), 2)
            _txt(img, f"Y {s['dyaw']:+6.1f}",  bx + 300, by, 0.8, (180, 220, 180), 2)
            _txt(img, "deg", bx + 440, by, 0.6, (120, 120, 120), 1)

            by += 38
            _txt(img, f"gripper {s['gripper']:5.1f}%  [trigger]",
                 bx, by, 0.8, (200, 200, 200), 2)
            _bar(img, bx + 310, by - 22, 280, 26, s["gripper"], trk_col)

            if i == 0:
                cv2.line(img, (638, 60), (638, OH - 20), (60, 60, 60), 1)

        footer = "Ctrl+C or ESC to stop  |  Restart to re-calibrate"
        if d["sim"]:
            footer += "   [SIMULATION - no robot motion]"
        _txt(img, footer, 20, OH - 20, 0.6, (80, 80, 80), 1)
    return img


def countdown_display(seconds, label):
    for i in range(seconds, 0, -1):
        _disp["countdown"] = i
        t_end = time.time() + 1.0
        while time.time() < t_end:
            if _cv2_ok:
                cv2.imshow("Remote Teleop 6DOF", render_frame())
                cv2.waitKey(1)
            print(f"\r  {label}  {i:2d} s ...  ", end="", flush=True)
            time.sleep(1 / 30.0)
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args        = parse_args()
    scale       = max(0.1, min(2.0, args.scale))
    rot_scale   = max(0.0, min(5.0, args.rot_scale))
    with_robot  = args.with_robot
    alpha       = float(np.clip(args.smoothing, 0.0, 1.0))
    invert_grip = bool(args.invert_gripper)
    ik_mode     = (args.joint_ik and not args.no_joint_ik) and with_robot
    rot_thresh  = math.radians(ROT_THRESH_DEG)

    P  = VR_TO_ROBOT
    Pt = VR_TO_ROBOT.T

    _disp["scale"]     = scale
    _disp["rot_scale"] = rot_scale
    _disp["sim"]       = not with_robot
    _disp["server"]    = f"{args.host}:{args.port}"

    # ── CAN port resolution & determine active arms ─────────────────────────
    left_can = right_can = None
    if with_robot:
        left_can  = args.left
        right_can = args.right
        if left_can is None and right_can is None:
            # Neither specified — try env file for both
            auto_l, auto_r = get_can_ports()
            left_can  = auto_l
            right_can = auto_r
        if not left_can and not right_can:
            print("  ERROR: No CAN ports specified. Use --left and/or --right.")
            sys.exit(1)

    # Which arms are active?
    use_left  = not with_robot or left_can is not None
    use_right = not with_robot or right_can is not None
    active_hands = []
    if use_left:  active_hands.append("left")
    if use_right: active_hands.append("right")

    if len(active_hands) == 2:
        arm_label = "both"
    else:
        arm_label = active_hands[0]
    _disp["arms"] = arm_label

    print()
    print("=" * 62)
    print("  6DOF TELEOP — PIPER HANGING ARMS")
    print("=" * 62)
    print(f"  Mode       : {'ROBOT' if with_robot else 'SIMULATION'}")
    print(f"  Arms       : {arm_label.upper()}")
    print(f"  Pos scale  : {scale}    Rot scale: {rot_scale}")
    print(f"  Hold: {HOLD_HZ} Hz   Update: every {UPDATE_INTERVAL}s")
    print(f"  Thresh pos : {MOVE_THRESH_CM} cm    rot: {ROT_THRESH_DEG}°")
    print(f"  Joint prot : {math.degrees(JOINT_LIMIT_MARGIN):.0f}° margin (reactive monitoring)")
    print(f"  IK mode    : {'LOCAL (joint ctrl, J2/J3 compensation)' if ik_mode else 'FIRMWARE (Cartesian ctrl)'}")
    print(f"  Smoothing  : {alpha}")
    if with_robot:
        print(f"  Speed      : {args.speed} %")
        if left_can:  print(f"  CAN LEFT   : {left_can}")
        if right_can: print(f"  CAN RIGHT  : {right_can}")
    print()

    # ── Start embedded HTTP relay server ──────────────────────────────────────
    _disp["phase"] = "starting"
    if _cv2_ok:
        cv2.imshow("Remote Teleop 6DOF", render_frame())
        cv2.waitKey(1)

    print(f"  Starting HTTP server on {args.host}:{args.port} ...")
    start_server(host=args.host, port=args.port, blocking=False)
    print(f"  Server ready. On Windows run:")
    print(f"    python vr_sender.py --server http://<THIS_IP>:{args.port}")
    print()

    # ── Wait for VR sender data (only required controllers) ──────────────────
    _disp["phase"] = "waiting"
    needed = ", ".join(h.upper() for h in active_hands)
    print(f"  Waiting for {needed} controller(s) ...", end=" ", flush=True)

    t_wait0 = time.time()
    while True:
        frame = read_latest_pose()
        if frame is not None:
            all_ok = True
            for hand in active_hands:
                _, _, _, st = extract_controller(frame, hand)
                if st != "OK":
                    all_ok = False
                    break
            if all_ok:
                break
        if time.time() - t_wait0 > 300.0:
            print("\n  TIMEOUT.")
            sys.exit(1)
        if _cv2_ok:
            cv2.imshow("Remote Teleop 6DOF", render_frame())
            cv2.waitKey(1)
        time.sleep(0.1)

    print("OK!")
    print()

    # ── Capture VR home reference pose (only active hands) ───────────────────
    _disp["phase"] = "calibrating"
    print(f"  {YELLOW}Hold controller(s) at robot home position.{RESET}")
    countdown_display(int(args.ref_secs) + 3, "Snapping home in")

    refs = capture_reference(secs=float(args.ref_secs), poll_interval=1.0 / HOLD_HZ,
                             hands=active_hands)
    vr_home_pos = {}
    vr_home_R   = {}
    for h in active_hands:
        vr_home_pos[h], vr_home_R[h] = refs[h]

    _disp["phase"] = "streaming"
    print(f"  {GREEN}VR home captured.{RESET}")
    print()

    # ── Connect robot arms & read actual home position ────────────────────────
    ctrls = {}   # "left" / "right" → PiperHangingController
    cmds  = {}   # "left" / "right" → ArmCommander

    # Default home from FK (simulation)
    T_home, _ = forward_kinematics(HOME_POSITION)
    fk_rpy = list(rotation_to_euler(T_home[:3, :3]))
    fk_pos = list(T_home[:3, 3])

    # FK home in world frame (for IK mode — avoids firmware/FK mismatch)
    ik_home_pos_w = np.array(fk_pos, dtype=np.float64)
    ik_home_R_w   = T_home[:3, :3].copy()

    robot_home = {}
    for h in active_hands:
        robot_home[h] = {
            "pos": fk_pos, "rpy": fk_rpy,
            "R": euler_to_rotation(*fk_rpy),
        }

    if with_robot:
        from piper_core import PiperHangingController

        can_map = {"left": left_can, "right": right_can}

        # Connect
        print("  Connecting arms ...")
        threads = []
        results = {}
        for h in active_hands:
            ctrl = PiperHangingController(can_port=can_map[h])
            ctrls[h] = ctrl
            results[h] = False
            def _conn(hand=h, c=ctrl):
                results[hand] = c.connect()
            t = threading.Thread(target=_conn)
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

        for h in active_hands:
            if not results[h]:
                print(f"  {h.upper()} arm failed to connect.")
                for oh in ctrls:
                    if oh != h:
                        try: ctrls[oh].shutdown()
                        except: pass
                sys.exit(1)
            ctrls[h].speed = args.speed

        # Home
        print("  Homing arms ...")
        threads = []
        for h in active_hands:
            t = threading.Thread(target=ctrls[h].go_home)
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

        # Read actual robot pose
        for h in active_hands:
            pos, rpy = ctrls[h].read_pose()
            robot_home[h] = {
                "pos": pos, "rpy": rpy,
                "R": euler_to_rotation(*rpy),
            }
            print(f"  {h.upper():5s} home: pos=({pos[0]*100:.2f}, {pos[1]*100:.2f}, {pos[2]*100:.2f}) cm"
                  f"  rpy=({math.degrees(rpy[0]):.1f}, {math.degrees(rpy[1]):.1f}, {math.degrees(rpy[2]):.1f})°")
        print()

        for h in active_hands:
            cmds[h] = ArmCommander(ctrls[h], h)

    # ── Per-arm state ────────────────────────────────────────────────────────
    I3 = np.eye(3, dtype=np.float64)
    # Position: committed deltas from robot home (cm)
    com_pos  = {h: [0.0, 0.0, 0.0] for h in active_hands}
    # Rotation: committed delta rotation MATRIX from home (identity = no rotation)
    com_R    = {h: I3.copy() for h in active_hands}
    # Smoothed position candidates (cm)
    smooth_pos = {h: [0.0, 0.0, 0.0] for h in active_hands}
    # Smoothed rotation delta matrix
    smooth_R   = {h: I3.copy() for h in active_hands}
    # Gripper
    com_grip  = {h: 100.0 for h in active_hands}
    grip_raw  = {h: 100.0 for h in active_hands}
    # Tracking status
    tracking  = {h: False for h in active_hands}
    # Joint limit protection state
    safe_pos   = {h: [0.0, 0.0, 0.0] for h in active_hands}
    safe_R     = {h: I3.copy() for h in active_hands}
    at_limit   = {h: False for h in active_hands}
    limit_info = {h: [] for h in active_hands}
    max_pos_d = MAX_POS_DELTA_CM
    # Joint IK state (seed with current joint positions after homing)
    last_ik_q = {}
    cmd_q     = {}   # smoothed joint commands (separate from IK seed)
    if ik_mode:
        for h in active_hands:
            q0 = ctrls[h].read_joints() if h in ctrls else list(HOME_POSITION)
            last_ik_q[h] = list(q0)
            cmd_q[h]     = list(q0)

    frame_num   = 0
    last_update = 0.0
    last_print  = 0.0

    sim_note = f"  {YELLOW}[SIMULATION]{RESET}" if not with_robot else ""
    print(f"  Streaming 6DOF ({arm_label}) — Ctrl+C to stop{sim_note}")
    print()

    # Print header
    cols = []
    for h in active_hands:
        tag = h[0].upper()
        cols.append(f"{tag+'_x':>7s} {tag+'_y':>7s} {tag+'_z':>7s}  "
                    f"{tag+'_R':>6s} {tag+'_P':>6s} {tag+'_Y':>6s}")
    grip_hdr = "  ".join(f"{h[0].upper()+'_gr':>5s}" for h in active_hands)
    print(f"  {'time':>8s}  {'  '.join(cols)}  {grip_hdr}  {'age':>5s}")
    print(f"  {'':->8s}  "
          + "  ".join(f"{'':->7s} {'':->7s} {'':->7s}  {'':->6s} {'':->6s} {'':->6s}" for _ in active_hands)
          + "  " + "  ".join(f"{'':->5s}" for _ in active_hands)
          + f"  {'':->5s}")

    # ── Main control loop ─────────────────────────────────────────────────────
    try:
        while True:
            t_start = time.time()

            frame = read_latest_pose()
            age_ms = frame.get("_age_ms", -1) if frame else -1
            _disp["age_ms"] = age_ms

            # ── Process each active controller ──────────────────────────────
            for hand in active_hands:
                pos, R, trig, status = extract_controller(frame, hand)
                ok = status == "OK"
                tracking[hand] = ok

                if ok:
                    # Position delta
                    raw_dp = remap((pos - vr_home_pos[hand]) * scale, AXIS_MAP)
                    smooth_pos[hand][0] = alpha * deadband(raw_dp[2] * 100, DEADBAND_CM) + (1 - alpha) * smooth_pos[hand][0]
                    smooth_pos[hand][1] = alpha * deadband(raw_dp[1] * 100, DEADBAND_CM) + (1 - alpha) * smooth_pos[hand][1]
                    smooth_pos[hand][2] = alpha * deadband(-raw_dp[0] * 100, DEADBAND_CM) + (1 - alpha) * smooth_pos[hand][2]

                    # Rotation: VR delta → robot frame → smooth as MATRIX
                    # (avoids all Euler singularity issues)
                    R_delta_vr  = rotation_delta(R, vr_home_R[hand])
                    R_delta_rob = P @ R_delta_vr @ Pt
                    if rot_scale != 1.0:
                        R_delta_rob = scale_rotation(R_delta_rob, rot_scale)
                    smooth_R[hand] = rotation_ema(R_delta_rob, smooth_R[hand], alpha)

                    # Gripper
                    t_val = 1.0 - trig if invert_grip else trig
                    grip_raw[hand] = float(np.clip(t_val * 100.0, 0.0, 100.0))

            frame_num += 1

            # ── Accept new targets (every UPDATE_INTERVAL) ──────────────────
            now = time.time()
            if now - last_update >= UPDATE_INTERVAL:
                last_update = now
                for hand in active_hands:
                    # Step 1: Check joint limits BEFORE accepting new deltas
                    near = False
                    if with_robot and hand in ctrls:
                        near, details = check_joint_limits(ctrls[hand])
                        at_limit[hand] = near
                        limit_info[hand] = details

                    # Step 2: Compute candidate new deltas via thresholding
                    new_pos = list(com_pos[hand])
                    for i in range(3):
                        if abs(smooth_pos[hand][i] - com_pos[hand][i]) >= MOVE_THRESH_CM:
                            new_pos[i] = smooth_pos[hand][i]

                    # Rotation threshold (angular distance between smoothed and committed)
                    if angular_distance(smooth_R[hand], com_R[hand]) >= rot_thresh:
                        new_R = smooth_R[hand].copy()
                    else:
                        new_R = com_R[hand]

                    # Step 3: Joint limit protection — when near limits,
                    # only allow deltas to shrink (move back toward home)
                    if near:
                        for i in range(3):
                            new_pos[i] = _clamp_toward_home(
                                new_pos[i], safe_pos[hand][i])
                        # Rotation: only allow if angular distance from identity decreases
                        if angular_distance(new_R, I3) > angular_distance(safe_R[hand], I3):
                            new_R = safe_R[hand].copy()
                    else:
                        # All joints safe — update safe checkpoint
                        safe_pos[hand] = list(new_pos)
                        safe_R[hand] = new_R.copy()

                    com_pos[hand] = new_pos
                    com_R[hand] = new_R

                    # Gripper
                    if abs(grip_raw[hand] - com_grip[hand]) >= GRIPPER_DEADBAND:
                        com_grip[hand] = grip_raw[hand]

            # ── Compute targets & send ────────────────────────────────────
            for hand in active_hands:
                # Position target (with backstop clamp)
                clamped_pos = [
                    max(-max_pos_d, min(max_pos_d, com_pos[hand][j]))
                    for j in range(3)
                ]
                tgt_pos = [
                    robot_home[hand]["pos"][j] + clamped_pos[j] / 100
                    for j in range(3)
                ]

                # Absolute rotation target via direct matrix composition
                R_target = com_R[hand] @ robot_home[hand]["R"]
                tgt_rpy = rotation_to_euler(R_target)

                if with_robot and hand in cmds:
                    if ik_mode:
                        # Local IK: compute target in FK world frame directly
                        # (avoids firmware FK ↔ our FK mismatch)
                        delta_m = np.array([clamped_pos[j] / 100.0 for j in range(3)])
                        pos_w = list(ik_home_pos_w + T_BASE_ROT @ delta_m)
                        R_delta_w = T_BASE_ROT @ com_R[hand] @ T_BASE_ROT
                        R_w = R_delta_w @ ik_home_R_w
                        q_sol = ik_solve(last_ik_q[hand], pos_w, R_w,
                                         max_iter=50, pos_tol=1e-3, rot_tol=0.02,
                                         margin=JOINT_LIMIT_MARGIN,
                                         joint_weights=IK_JOINT_WEIGHTS)
                        if q_sol is not None:
                            last_ik_q[hand] = list(q_sol)  # unsmoothed seed
                            for i in range(6):
                                cmd_q[hand][i] += IK_JOINT_ALPHA[i] * (q_sol[i] - cmd_q[hand][i])
                            cmds[hand].send_joints(cmd_q[hand])
                    else:
                        # Firmware IK: send Cartesian target
                        cmds[hand].send(
                            tgt_pos[0], tgt_pos[1], tgt_pos[2],
                            tgt_rpy[0], tgt_rpy[1], tgt_rpy[2],
                        )
                    cmds[hand].gripper(com_grip[hand] / 100.0 * GRIPPER_MAX_MM)

            # ── Display at DISPLAY_HZ ───────────────────────────────────────
            if now - last_print >= 1.0 / DISPLAY_HZ:
                last_print = now

                parts = []
                for hand in active_hands:
                    cp = [max(-max_pos_d, min(max_pos_d, com_pos[hand][j])) for j in range(3)]
                    sp = [robot_home[hand]["pos"][j] + cp[j] / 100 for j in range(3)]
                    # Extract Euler from committed rotation matrix for display
                    drpy_raw = rotation_to_euler(com_R[hand])
                    drpy_disp = normalize_delta_euler(*drpy_raw)
                    d_rpy = [math.degrees(x) for x in drpy_disp]
                    parts.append(
                        f"{sp[0]*100:+7.2f} {sp[1]*100:+7.2f} {sp[2]*100:+7.2f}  "
                        f"{d_rpy[0]:+6.1f} {d_rpy[1]:+6.1f} {d_rpy[2]:+6.1f}"
                    )

                grip_parts = [f"{com_grip[h]:5.1f}" for h in active_hands]

                # Joint limit indicator
                limit_str = ""
                for hand in active_hands:
                    if at_limit.get(hand, False):
                        jstr = ",".join(f"J{j}{d}" for j, d in limit_info.get(hand, []))
                        limit_str += f" {RED}LIMIT({hand[0].upper()}:{jstr}){RESET}"

                print(f"  {now - t_wait0:7.1f}s  "
                      f"{'  '.join(parts)}  "
                      f"{'  '.join(grip_parts)}  "
                      f"{age_ms:5.0f}ms{limit_str}")

                _disp["frame"] = frame_num
                for hand in active_hands:
                    drpy_raw = rotation_to_euler(com_R[hand])
                    drpy_disp = normalize_delta_euler(*drpy_raw)
                    d_rpy = [math.degrees(x) for x in drpy_disp]
                    _disp[hand] = dict(
                        tracking=tracking[hand],
                        dx=com_pos[hand][0], dy=com_pos[hand][1], dz=com_pos[hand][2],
                        droll=d_rpy[0], dpitch=d_rpy[1], dyaw=d_rpy[2],
                        gripper=com_grip[hand])

                if _cv2_ok:
                    cv2.imshow("Remote Teleop 6DOF", render_frame())
                    if cv2.waitKey(1) == 27:
                        break

            # ── Rate limit (~40 Hz) ─────────────────────────────────────────
            elapsed = time.time() - t_start
            sleep = 0.025 - elapsed
            if sleep > 0:
                time.sleep(sleep)

    except KeyboardInterrupt:
        print("\n\n  Stopped.")

    finally:
        for h in cmds:
            cmds[h].stop()

        if with_robot and ctrls:
            print("  Relaxing arms ...", end=" ", flush=True)
            try:
                threads = [threading.Thread(target=ctrls[h].go_relax) for h in ctrls]
                for t in threads: t.start()
                for t in threads: t.join()
                print("done")
            except Exception as e:
                print(f"skipped ({e})")

            print("  Shutting down ...", end=" ", flush=True)
            try:
                for h in ctrls:
                    ctrls[h].shutdown()
                print("done")
            except Exception as e:
                print(f"skipped ({e})")

        if _cv2_ok:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
