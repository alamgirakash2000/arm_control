#!/usr/bin/env python3
"""
Remote Teleop 6DOF — Piper Hanging Arms (single or dual)
=========================================================
Full 6-DOF control: position (X, Y, Z) AND orientation (roll, pitch, yaw).

Supports single-arm mode:
    python teleop.py --with_robot --left can0          # left arm only
    python teleop.py --with_robot --right can1         # right arm only
    python teleop.py --with_robot --left can0 --right can1  # both

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
import json
import math
import os
import select
import sys
import termios
import time
import threading
import tty

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
    J6_PHYSICAL_OFFSET,
)

from relay_server import pose_store, start_server

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.*=false"
try:
    import cv2
    _cv2_ok = True
except ImportError:
    _cv2_ok = False

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    _arrow_ok = True
except ImportError:
    _arrow_ok = False

import subprocess as _sp

def _say(text, wait=False):
    """Speak text aloud using spd-say. wait=True blocks until done."""
    try:
        args = ["spd-say", "-r", "10"]
        if wait:
            args.append("-w")
        args.append(text)
        _sp.Popen(args, stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
    except Exception:
        pass


# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_SCALE      = 1.0
DEFAULT_ROT_SCALE  = 1.0
DEFAULT_SPEED      = 70
DEFAULT_SMOOTHING  = 0.08     # lower = smoother (was 0.15)
GRIPPER_MAX_MM     = 70.0
GRIPPER_DEADBAND   = 2.0      # %
DEADBAND_CM        = 0.5      # cm — per-axis EMA deadband (was 0.3)
MOVE_THRESH_CM     = 0.25     # cm — per-axis position threshold (was 0.15)
ROT_THRESH_DEG     = 1.0      # deg — rotation threshold (was 0.5)
MAX_POS_STEP_CM    = 1.0      # cm — max position change per update (velocity cap)
MAX_ROT_STEP_DEG   = 2.5      # deg — max rotation change per update (velocity cap)
UPDATE_INTERVAL    = 0.025    # seconds between accepting new targets (40 Hz)
HOLD_HZ            = 30       # Hz — rate to resend current target
DISPLAY_HZ         = 10        # terminal print rate
REF_SECS           = 2.5      # seconds to average for home pose capture
STALE_MS           = 500.0    # pose data older than this = tracking lost
MAX_POS_DELTA_CM   = 100.0    # generous backstop (joint limits are the real safety)
MAX_ROT_DELTA_DEG  = 180.0    # generous backstop (joint limits are the real safety)
# Per-joint limit margins (J1-J6). J4=10°, J5=5°, others=10°.
JOINT_LIMIT_MARGINS = [math.radians(d) for d in [10.0, 10.0, 10.0, 10.0, 5.0, 10.0]]
# IK solver margins: larger for J5 (only 180° range) to keep IK solutions away from the edge.
IK_JOINT_MARGINS    = [math.radians(d) for d in [10.0, 10.0, 10.0, 10.0, 15.0, 10.0]]

# IK joint-space smoothing: per-joint EMA rate (0=frozen, 1=instant)
# J4 (forearm twist) tracks slower to prevent slamming into limits
IK_JOINT_ALPHA   = [0.08, 0.08, 0.08, 0.06, 0.06, 0.08]  # J4+J5 track slower (was 0.08)
# IK joint weights: higher = joint moves less (penalizes J4+J5 to prevent limit-hitting)
IK_JOINT_WEIGHTS = [1.0, 1.0, 1.0, 4.0, 4.0, 1.0]        # J4+J5: 4x penalty (was 2x/1x)

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
CYAN   = "\033[36m"
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


def compute_yaw_fix(R_home):
    """Compute a Y-axis rotation matrix that undoes the tracking-space yaw.

    SteamVR's tracking universe can be oriented arbitrarily (e.g. 180° rotated
    when the Quest reconnects facing a different direction).  This extracts the
    yaw component from the home rotation and returns a matrix that undoes it,
    so position and rotation deltas are tracking-space invariant.

    With "normal" tracking (yaw ≈ 0) this returns ~identity → no effect.
    """
    rx, rz = float(R_home[0, 0]), float(R_home[2, 0])
    yaw = math.atan2(rz, rx)  # = -θ where θ is tracking space yaw
    c, s = math.cos(yaw), math.sin(yaw)  # R_y(-θ) undoes the yaw
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def remap(pos_delta_m, axis_map):
    return np.array([pos_delta_m[ax] * sign for ax, sign in axis_map])


def check_joint_limits(ctrl, margins=JOINT_LIMIT_MARGINS):
    """Read robot joints and check if any are near limits.

    Args:
        margins: per-joint angular margins in radians

    Returns (near_limit, near_details) where near_details is a list of
    (joint_number_1based, 'lo'|'hi') tuples.
    """
    try:
        joints = ctrl.read_joints()
    except Exception:
        return False, []
    near = []
    for i in range(6):
        m = margins[i]
        if joints[i] <= JOINT_LOWER[i] + m:
            near.append((i + 1, 'lo'))
        elif joints[i] >= JOINT_UPPER[i] - m:
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

    def clear(self):
        """Stop sending any commands (silence the background thread)."""
        with self._lock:
            self._target = None
            self._joint_target = None
            self._grip_mm = None

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
                    q_physical = list(jtgt)
                    q_physical[5] += J6_PHYSICAL_OFFSET  # logical → firmware
                    jcmds = [round(q * RAD_TO_MDEG) for q in q_physical]
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
    # Recording (on by default — use --no-record to disable)
    p.add_argument("--record",         default="./data/tool_good", metavar="DIR",
                   help="Dataset directory (default: ./data/tool_good)")
    p.add_argument("--no-record",      action="store_true",
                   help="Disable recording")
    p.add_argument("--rec-fps",        default=20, type=int,
                   help="Recording FPS (default: 20)")
    p.add_argument("--cameras",        default="auto", metavar="IDS",
                   help="Camera IDs (default: auto-detect)")
    p.add_argument("--task",           default="teleoperation demo",
                   help="Task description for dataset metadata")
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
    _say(f"{label} {seconds} seconds")
    for i in range(seconds, 0, -1):
        _disp["countdown"] = i
        if i <= 3:
            _say(str(i))
        t_end = time.time() + 1.0
        while time.time() < t_end:
            if _cv2_ok:
                cv2.imshow("Remote Teleop 6DOF", render_frame())
                cv2.waitKey(1)
            print(f"\r  {label}  {i:2d} s ...  ", end="", flush=True)
            time.sleep(1 / 30.0)
    print()


# ── Recording (LeRobot v2.1) ──────────────────────────────────────────────────

def _extract_buttons(frame, hand):
    """Return button dict from VR frame data."""
    if frame is None:
        return {}
    c = frame.get(hand)
    if c is None:
        return {}
    return c.get("buttons", {})


class EpisodeBuffer:
    """Accumulate timestep data for one recording episode."""

    def __init__(self):
        self.rows = []
        self.video_frames = {}   # {cam_name: [bgr_frame, ...]}
        self.t0 = None

    def add(self, joints, eef_pos, eef_rot_flat, eef_euler, gripper,
            camera_frames):
        now = time.time()
        if self.t0 is None:
            self.t0 = now
        ts = now - self.t0

        # Store raw observations only — actions are computed at save time
        # with +1 timestep shift (action[t] = observation[t+1])
        self.rows.append({
            "timestamp": ts,
            "obs_state": list(joints) + [gripper],
            "obs_eef_pos": list(eef_pos),
            "obs_eef_rot": list(eef_rot_flat),
            "obs_eef_euler": list(eef_euler),
        })
        for name, fr in camera_frames.items():
            if name not in self.video_frames:
                self.video_frames[name] = []
            self.video_frames[name].append(fr)

    def __len__(self):
        return len(self.rows)


class DatasetWriter:
    """Write episodes to a LeRobot v2.1 directory."""

    def __init__(self, dataset_dir, fps, task, camera_names):
        self.dir = os.path.abspath(dataset_dir)
        self.fps = fps
        self.task = task
        self.cam_names = camera_names
        self.ep_count = 0
        self.total_frames = 0

        os.makedirs(os.path.join(self.dir, "data", "chunk-000"), exist_ok=True)
        os.makedirs(os.path.join(self.dir, "meta"), exist_ok=True)
        for c in camera_names:
            os.makedirs(os.path.join(self.dir, "videos", "chunk-000",
                                     f"observation.images.{c}"), exist_ok=True)
        info_path = os.path.join(self.dir, "meta", "info.json")
        if os.path.exists(info_path):
            with open(info_path) as f:
                info = json.load(f)
            self.ep_count = info.get("total_episodes", 0)
            self.total_frames = info.get("total_frames", 0)

    def save(self, buf):
        if not _arrow_ok:
            print(f"  {RED}pyarrow not installed — cannot save{RESET}")
            return None
        ep = self.ep_count
        n = len(buf)
        cols = {
            "timestamp": [], "episode_index": [], "frame_index": [],
            "index": [], "next.done": [], "task_index": [],
            "observation.state": [], "observation.eef_pos": [],
            "observation.eef_rot": [], "observation.eef_euler": [],
            "action.abs_joint": [], "action.delta_joint": [],
            "action.abs_eef": [], "action.delta_eef": [],
        }
        for i, row in enumerate(buf.rows):
            cols["timestamp"].append(row["timestamp"])
            cols["episode_index"].append(ep)
            cols["frame_index"].append(i)
            cols["index"].append(self.total_frames + i)
            cols["next.done"].append(i == n - 1)
            cols["task_index"].append(0)
            # Observations: current frame
            cols["observation.state"].append(row["obs_state"])
            cols["observation.eef_pos"].append(row["obs_eef_pos"])
            cols["observation.eef_rot"].append(row["obs_eef_rot"])
            cols["observation.eef_euler"].append(row["obs_eef_euler"])
            # Actions: shifted by +1 — action[t] = observation[t+1]
            # This is the standard convention for imitation learning:
            # "given state t, predict where to go next"
            if i < n - 1:
                nxt = buf.rows[i + 1]
                cur_s = row["obs_state"]
                nxt_s = nxt["obs_state"]
                cols["action.abs_joint"].append(nxt_s)
                cols["action.delta_joint"].append(
                    [nxt_s[j] - cur_s[j] for j in range(7)])
                nxt_eef = (list(nxt["obs_eef_pos"])
                           + list(nxt["obs_eef_euler"]) + [nxt_s[6]])
                cols["action.abs_eef"].append(nxt_eef)
                cur_eef_p = row["obs_eef_pos"]
                cur_eef_e = row["obs_eef_euler"]
                nxt_eef_p = nxt["obs_eef_pos"]
                nxt_eef_e = nxt["obs_eef_euler"]
                cols["action.delta_eef"].append(
                    [nxt_eef_p[j] - cur_eef_p[j] for j in range(3)]
                    + [nxt_eef_e[j] - cur_eef_e[j] for j in range(3)]
                    + [nxt_s[6] - cur_s[6]])
            else:
                # Last frame: hold position (zero delta)
                cols["action.abs_joint"].append(row["obs_state"])
                cols["action.delta_joint"].append([0.0] * 7)
                cols["action.abs_eef"].append(
                    list(row["obs_eef_pos"]) + list(row["obs_eef_euler"])
                    + [row["obs_state"][6]])
                cols["action.delta_eef"].append([0.0] * 7)
        for cam in self.cam_names:
            vp = f"videos/chunk-000/observation.images.{cam}/episode_{ep:06d}.mp4"
            cols[f"observation.images.{cam}"] = [
                {"path": vp, "timestamp": row["timestamp"]} for row in buf.rows
            ]

        arrays = {}
        for key, vals in cols.items():
            if key.startswith("observation.images."):
                arrays[key] = pa.StructArray.from_arrays(
                    [pa.array([v["path"] for v in vals], type=pa.string()),
                     pa.array([v["timestamp"] for v in vals], type=pa.float32())],
                    names=["path", "timestamp"])
            elif isinstance(vals[0], list):
                arrays[key] = pa.array(vals, type=pa.list_(pa.float32()))
            elif isinstance(vals[0], bool):
                arrays[key] = pa.array(vals, type=pa.bool_())
            elif isinstance(vals[0], int):
                arrays[key] = pa.array(vals, type=pa.int64())
            else:
                arrays[key] = pa.array(vals, type=pa.float32())
        pq.write_table(pa.table(arrays),
                        os.path.join(self.dir, "data", "chunk-000",
                                     f"episode_{ep:06d}.parquet"))

        for cam in self.cam_names:
            frames = buf.video_frames.get(cam, [])
            if frames and _cv2_ok:
                vpath = os.path.join(self.dir, "videos", "chunk-000",
                                     f"observation.images.{cam}",
                                     f"episode_{ep:06d}.mp4")
                h, w = frames[0].shape[:2]
                wr = cv2.VideoWriter(vpath, cv2.VideoWriter_fourcc(*'mp4v'),
                                     self.fps, (w, h))
                for fr in frames:
                    wr.write(fr)
                wr.release()

        self.ep_count += 1
        self.total_frames += n
        self._write_meta(ep, n)
        return ep

    def _write_meta(self, ep_idx, ep_len):
        meta = os.path.join(self.dir, "meta")
        features = {}
        for name, shape, nms in [
            ("observation.state", [7], ["j1","j2","j3","j4","j5","j6","gripper"]),
            ("observation.eef_pos", [3], ["x","y","z"]),
            ("observation.eef_rot", [9], ["r00","r01","r02","r10","r11","r12","r20","r21","r22"]),
            ("observation.eef_euler", [3], ["roll","pitch","yaw"]),
            ("action.abs_joint", [7], ["j1","j2","j3","j4","j5","j6","gripper"]),
            ("action.delta_joint", [7], ["dj1","dj2","dj3","dj4","dj5","dj6","dgripper"]),
            ("action.abs_eef", [7], ["x","y","z","roll","pitch","yaw","gripper"]),
            ("action.delta_eef", [7], ["dx","dy","dz","droll","dpitch","dyaw","dgripper"]),
        ]:
            features[name] = {"dtype": "float32", "shape": shape, "names": nms}
        for c in self.cam_names:
            features[f"observation.images.{c}"] = {
                "dtype": "video", "shape": [480, 640, 3],
                "names": ["height","width","channel"],
                "info": {"video.fps": self.fps, "video.codec": "mp4v"},
            }
        for k, dt, sh in [("episode_index","int64",[1]),("frame_index","int64",[1]),
                           ("timestamp","float32",[1]),("index","int64",[1]),
                           ("next.done","bool",[1]),("task_index","int64",[1])]:
            features[k] = {"dtype": dt, "shape": sh}
        info = {
            "codebase_version": "v2.1", "robot_type": "piper_hanging",
            "fps": self.fps, "total_episodes": self.ep_count,
            "total_frames": self.total_frames, "total_tasks": 1,
            "total_videos": self.ep_count * len(self.cam_names),
            "splits": {"train": f"0:{self.ep_count}"}, "features": features,
        }
        with open(os.path.join(meta, "info.json"), "w") as f:
            json.dump(info, f, indent=2)
        with open(os.path.join(meta, "tasks.jsonl"), "w") as f:
            f.write(json.dumps({"task_index": 0, "task": self.task}) + "\n")
        with open(os.path.join(meta, "episodes.jsonl"), "a") as f:
            f.write(json.dumps({"episode_index": ep_idx, "tasks": [self.task],
                                "length": ep_len}) + "\n")


class CameraCapture:
    """Manage USB cameras for recording."""

    def __init__(self, cam_ids):
        self.caps = {}
        self.names = []
        for cid in cam_ids:
            cap = cv2.VideoCapture(cid)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                name = f"cam_{cid}"
                self.caps[name] = cap
                self.names.append(name)

    def read_all(self):
        out = {}
        for name, cap in self.caps.items():
            ret, frame = cap.read()
            if ret:
                out[name] = frame
        return out

    def release(self):
        for cap in self.caps.values():
            cap.release()


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
    print(f"  Joint prot : [{', '.join(f'{math.degrees(m):.0f}' for m in JOINT_LIMIT_MARGINS)}]° per-joint margin")
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
    _say("Controller connected")
    print()

    # ── Capture VR home reference pose (only active hands) ───────────────────
    _disp["phase"] = "calibrating"
    _say("Hold controller at robot home position")
    print(f"  {YELLOW}Hold controller(s) at robot home position.{RESET}")
    countdown_display(int(args.ref_secs) + 3, "Snapping home in")

    refs = capture_reference(secs=float(args.ref_secs), poll_interval=1.0 / HOLD_HZ,
                             hands=active_hands)
    vr_home_pos = {}
    vr_home_R   = {}
    vr_yaw_fix  = {}
    for h in active_hands:
        vr_home_pos[h], vr_home_R[h] = refs[h]
        vr_yaw_fix[h] = compute_yaw_fix(vr_home_R[h])

    _disp["phase"] = "streaming"
    _say("Calibration complete")
    print(f"  {GREEN}VR home captured.{RESET}")
    for h in active_hands:
        p = vr_home_pos[h]
        R = vr_home_R[h]
        rx, rz = float(R[0, 0]), float(R[2, 0])
        yaw_deg = math.degrees(math.atan2(rz, rx))
        print(f"  {h.upper()} VR ref pos: [{p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f}]")
        print(f"  {h.upper()} tracking yaw: {yaw_deg:+.1f}°  "
              f"({'normal' if abs(yaw_deg) < 45 else 'rotated — auto-correcting'})")
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

    # ── Recording setup ────────────────────────────────────────────────────
    rec_enabled = not args.no_record
    rec_writer = None
    rec_cameras = None
    rec_hand = active_hands[-1]  # use right controller for buttons (or only one)

    if rec_enabled:
        if not _arrow_ok:
            print(f"  {RED}ERROR: pyarrow required for recording. "
                  f"pip install pyarrow{RESET}")
            sys.exit(1)
        # Auto-detect camera: try indices 0-4, use the first one that opens
        if args.cameras == "auto":
            cam_ids = []
            if _cv2_ok:
                for test_id in range(5):
                    cap = cv2.VideoCapture(test_id)
                    if cap.isOpened():
                        cam_ids.append(test_id)
                        cap.release()
                        break  # only one camera
                    cap.release()
        else:
            cam_ids = [int(x) for x in args.cameras.split(",") if x.strip()]
        if _cv2_ok and cam_ids:
            rec_cameras = CameraCapture(cam_ids)
            print(f"  Cameras    : {rec_cameras.names}")
        else:
            rec_cameras = CameraCapture([])
            if _cv2_ok:
                print(f"  {YELLOW}No camera found — recording without video{RESET}")
        rec_writer = DatasetWriter(args.record, args.rec_fps, args.task,
                                   rec_cameras.names)
        print(f"  Recording  : {args.record}")
        print(f"  Rec FPS    : {args.rec_fps}")
        print(f"  Task       : {args.task}")
        print(f"  Episodes   : {rec_writer.ep_count} existing")
        print(f"  Buttons    : {rec_hand.upper()} controller "
              f"(X/A=start/discard  Y/B=stop/save)")
        print()

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

    # Recording state machine
    REC_IDLE, REC_RECORDING = "IDLE", "RECORDING"
    rec_state = REC_IDLE
    rec_buf = None
    rec_last_sample = 0.0
    rec_interval = 1.0 / args.rec_fps if rec_enabled else 1.0

    sim_note = f"  {YELLOW}[SIMULATION]{RESET}" if not with_robot else ""
    print(f"  Streaming 6DOF ({arm_label}) — Ctrl+C to stop{sim_note}")
    if rec_enabled:
        print(f"  {CYAN}Recording controls (press in terminal):{RESET}")
        print(f"    {GREEN}R{RESET} = start recording    {YELLOW}S{RESET} = stop + save")
        print(f"    {RED}D{RESET} = discard             ESC = quit")
    print()

    # ── Terminal raw mode for single-keypress input ────────────────────────────
    _old_term = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    def _get_key():
        """Non-blocking single keypress from terminal. Returns key code or -1."""
        if select.select([sys.stdin], [], [], 0)[0]:
            ch = sys.stdin.read(1)
            return ord(ch)
        return -1

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
                    # Position delta (yaw-corrected for tracking-space invariance)
                    dp_vr = (pos - vr_home_pos[hand]) * scale
                    dp_fixed = vr_yaw_fix[hand] @ dp_vr
                    raw_dp = remap(dp_fixed, AXIS_MAP)
                    smooth_pos[hand][0] = alpha * deadband(raw_dp[2] * 100, DEADBAND_CM) + (1 - alpha) * smooth_pos[hand][0]
                    smooth_pos[hand][1] = alpha * deadband(raw_dp[1] * 100, DEADBAND_CM) + (1 - alpha) * smooth_pos[hand][1]
                    smooth_pos[hand][2] = alpha * deadband(-raw_dp[0] * 100, DEADBAND_CM) + (1 - alpha) * smooth_pos[hand][2]

                    # Rotation: yaw-corrected VR delta → robot frame → smooth
                    R_delta_vr  = rotation_delta(R, vr_home_R[hand])
                    Yf = vr_yaw_fix[hand]
                    R_delta_fixed = Yf @ R_delta_vr @ Yf.T
                    R_delta_rob = P @ R_delta_fixed @ Pt
                    if rot_scale != 1.0:
                        R_delta_rob = scale_rotation(R_delta_rob, rot_scale)
                    smooth_R[hand] = rotation_ema(R_delta_rob, smooth_R[hand], alpha)

                    # Gripper
                    t_val = 1.0 - trig if invert_grip else trig
                    grip_raw[hand] = float(np.clip(t_val * 100.0, 0.0, 100.0))

            frame_num += 1

            # ── Recording state machine (keyboard: R=record, S=save, D=discard) ──
            if rec_enabled and rec_state == REC_RECORDING:
                now_rec = time.time()
                if now_rec - rec_last_sample >= rec_interval:
                    rec_last_sample = now_rec
                    try:
                        rj = ctrls[rec_hand].read_joints() if rec_hand in ctrls else list(HOME_POSITION)
                        T_ee, _ = forward_kinematics(rj)
                        eef_p = T_ee[:3, 3].tolist()
                        eef_R = T_ee[:3, :3]
                        eef_rot_flat = eef_R.flatten().tolist()
                        eef_eul = list(rotation_to_euler(eef_R))
                        g_val = com_grip[rec_hand] / 100.0
                        cam_frames = rec_cameras.read_all() if rec_cameras else {}
                        rec_buf.add(
                            rj, eef_p, eef_rot_flat, eef_eul, g_val,
                            cam_frames,
                        )
                    except Exception as e:
                        if frame_num % 200 == 0:
                            print(f"\n  [REC WARN] {e}")

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

                    # Velocity cap: limit how much target can jump in one update
                    for i in range(3):
                        delta = new_pos[i] - com_pos[hand][i]
                        if abs(delta) > MAX_POS_STEP_CM:
                            new_pos[i] = com_pos[hand][i] + math.copysign(MAX_POS_STEP_CM, delta)
                    max_rot_step_rad = math.radians(MAX_ROT_STEP_DEG)
                    ang_dist = angular_distance(new_R, com_R[hand])
                    if ang_dist > max_rot_step_rad and ang_dist > 1e-6:
                        R_rel = new_R @ com_R[hand].T
                        R_rel_capped = scale_rotation(R_rel, max_rot_step_rad / ang_dist)
                        new_R = R_rel_capped @ com_R[hand]

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
                        delta_m = np.array([clamped_pos[j] / 100.0 for j in range(3)])
                        pos_w = list(ik_home_pos_w + T_BASE_ROT @ delta_m)
                        R_delta_w = T_BASE_ROT @ com_R[hand] @ T_BASE_ROT
                        R_w = R_delta_w @ ik_home_R_w
                        q_sol = ik_solve(last_ik_q[hand], pos_w, R_w,
                                         max_iter=50, pos_tol=1e-3, rot_tol=0.02,
                                         joint_margins=IK_JOINT_MARGINS,
                                         joint_weights=IK_JOINT_WEIGHTS)
                        if q_sol is not None:
                            last_ik_q[hand] = list(q_sol)
                            for i in range(6):
                                cmd_q[hand][i] += IK_JOINT_ALPHA[i] * (q_sol[i] - cmd_q[hand][i])
                        # Hard clamp: NEVER send joints past per-joint margin
                        for i in range(6):
                            lo = JOINT_LOWER[i] + JOINT_LIMIT_MARGINS[i]
                            hi = JOINT_UPPER[i] - JOINT_LIMIT_MARGINS[i]
                            cmd_q[hand][i] = max(lo, min(hi, cmd_q[hand][i]))
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

                # Build compact status for cv2 overlay (detailed data stays on screen only)
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
                    # Show live camera feed
                    if rec_cameras and rec_cameras.caps:
                        cam_frames = rec_cameras.read_all()
                        for cname, cframe in cam_frames.items():
                            label = "REC" if rec_state == REC_RECORDING else "LIVE"
                            color = (0, 0, 255) if rec_state == REC_RECORDING else (0, 255, 0)
                            cv2.putText(cframe, f"{label} {cname}",
                                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.7, color, 2)
                            cv2.imshow(cname, cframe)
                    cv2.waitKey(1)

            # ── Recording status on terminal ───────────────────────────────
            if rec_enabled and rec_state == REC_RECORDING and rec_buf:
                n = len(rec_buf)
                dur = rec_buf.rows[-1]["timestamp"] if rec_buf.rows else 0
                print(f"\r  {RED}● RECORDING{RESET}  "
                      f"{n} frames / {dur:.1f}s  "
                      f"(S=save  D=discard)        ",
                      end="", flush=True)

            # ── Keyboard input from terminal (R/S/D/ESC) ──────────────────
            key = _get_key()
            if key == 27:  # ESC
                break
            if rec_enabled:
                if key == ord('r') and rec_state == REC_IDLE:
                    # Home robot first — sync ArmCommander before go_home
                    # so the background thread doesn't fight the homing
                    if with_robot and rec_hand in ctrls:
                        _say("Homing robot")
                        print(f"\n  {YELLOW}Homing robot before recording...{RESET}")
                        cmds[rec_hand].clear()
                        try:
                            ctrls[rec_hand].go_home()
                        except Exception:
                            pass
                        if ik_mode:
                            cmd_q[rec_hand] = list(HOME_POSITION)
                            last_ik_q[rec_hand] = list(HOME_POSITION)
                    # 5-second countdown + recalibrate before recording
                    _say("Prepare for calibration in 5 seconds")
                    for sec in range(5, 0, -1):
                        print(f"\r  {YELLOW}Calibrating in {sec}s — "
                              f"hold controller at home ...{RESET}   ",
                              end="", flush=True)
                        if sec <= 3:
                            _say(str(sec))
                        time.sleep(1)
                    print()
                    _say("Calibrating")
                    try:
                        refs = capture_reference(
                            secs=1.5, poll_interval=1.0 / HOLD_HZ,
                            hands=active_hands)
                        for h in active_hands:
                            vr_home_pos[h], vr_home_R[h] = refs[h]
                            vr_yaw_fix[h] = compute_yaw_fix(vr_home_R[h])
                            smooth_pos[h] = [0.0, 0.0, 0.0]
                            smooth_R[h] = I3.copy()
                            com_pos[h] = [0.0, 0.0, 0.0]
                            com_R[h] = I3.copy()
                            safe_pos[h] = [0.0, 0.0, 0.0]
                            safe_R[h] = I3.copy()
                            if ik_mode and h in ctrls:
                                q0 = ctrls[h].read_joints()
                                last_ik_q[h] = list(q0)
                                cmd_q[h] = list(q0)
                        print(f"  {GREEN}Recalibrated.{RESET}")
                    except Exception as e:
                        print(f"  {RED}Recalibration failed: {e}{RESET}")
                    rec_buf = EpisodeBuffer()
                    rec_last_sample = 0.0
                    rec_state = REC_RECORDING
                    _say("Recording started")
                    print(f"  {GREEN}● REC started "
                          f"(ep {rec_writer.ep_count}){RESET}")
                elif key == ord('s') and rec_state == REC_RECORDING:
                    # Stop + save
                    _say("Recording stopped")
                    n = len(rec_buf) if rec_buf else 0
                    dur = rec_buf.rows[-1]["timestamp"] if n > 0 else 0
                    print(f"\n  {YELLOW}■ Saving... "
                          f"({n} frames, {dur:.1f}s){RESET}")
                    if with_robot and rec_hand in ctrls:
                        cmds[rec_hand].clear()
                        try:
                            ctrls[rec_hand].go_home()
                        except Exception:
                            pass
                        if ik_mode:
                            cmd_q[rec_hand] = list(HOME_POSITION)
                            last_ik_q[rec_hand] = list(HOME_POSITION)
                    ep = rec_writer.save(rec_buf)
                    _say(f"Recording saved. Episode {ep}")
                    print(f"  {GREEN}✓ Saved episode {ep}{RESET}")
                    rec_buf = None
                    rec_state = REC_IDLE
                    # 15-second countdown before recalibration
                    _say("Recalibration starting in 15 seconds")
                    for sec in range(15, 0, -1):
                        print(f"\r  {YELLOW}Recalibrating in {sec}s — "
                              f"hold controller at home ...{RESET}   ",
                              end="", flush=True)
                        if sec <= 5:
                            _say(str(sec))
                        time.sleep(1)
                    print()
                    _say("Calibrating")
                    try:
                        refs = capture_reference(
                            secs=1.5, poll_interval=1.0 / HOLD_HZ,
                            hands=active_hands)
                        for h in active_hands:
                            vr_home_pos[h], vr_home_R[h] = refs[h]
                            vr_yaw_fix[h] = compute_yaw_fix(vr_home_R[h])
                            smooth_pos[h] = [0.0, 0.0, 0.0]
                            smooth_R[h] = I3.copy()
                            com_pos[h] = [0.0, 0.0, 0.0]
                            com_R[h] = I3.copy()
                            safe_pos[h] = [0.0, 0.0, 0.0]
                            safe_R[h] = I3.copy()
                            if ik_mode and h in ctrls:
                                q0 = ctrls[h].read_joints()
                                last_ik_q[h] = list(q0)
                                cmd_q[h] = list(q0)
                        _say("Recalibrated. Ready")
                        print(f"  {GREEN}Recalibrated. Ready.{RESET}")
                    except Exception as e:
                        print(f"  {RED}Recalibration failed: {e}{RESET}")
                elif key == ord('d') and rec_state == REC_RECORDING:
                    # Discard + home + recalibrate
                    _say("Recording discarded")
                    n = len(rec_buf) if rec_buf else 0
                    rec_buf = None
                    rec_state = REC_IDLE
                    print(f"\n  {RED}✗ Discarded ({n} frames){RESET}")
                    if with_robot and rec_hand in ctrls:
                        cmds[rec_hand].clear()
                        try:
                            ctrls[rec_hand].go_home()
                        except Exception:
                            pass
                        if ik_mode:
                            cmd_q[rec_hand] = list(HOME_POSITION)
                            last_ik_q[rec_hand] = list(HOME_POSITION)
                    # 15-second countdown before recalibration
                    _say("Recalibration starting in 15 seconds")
                    for sec in range(15, 0, -1):
                        print(f"\r  {YELLOW}Recalibrating in {sec}s — "
                              f"hold controller at home ...{RESET}   ",
                              end="", flush=True)
                        if sec <= 5:
                            _say(str(sec))
                        time.sleep(1)
                    print()
                    _say("Calibrating")
                    try:
                        refs = capture_reference(
                            secs=1.5, poll_interval=1.0 / HOLD_HZ,
                            hands=active_hands)
                        for h in active_hands:
                            vr_home_pos[h], vr_home_R[h] = refs[h]
                            vr_yaw_fix[h] = compute_yaw_fix(vr_home_R[h])
                            smooth_pos[h] = [0.0, 0.0, 0.0]
                            smooth_R[h] = I3.copy()
                            com_pos[h] = [0.0, 0.0, 0.0]
                            com_R[h] = I3.copy()
                            safe_pos[h] = [0.0, 0.0, 0.0]
                            safe_R[h] = I3.copy()
                            if ik_mode and h in ctrls:
                                q0 = ctrls[h].read_joints()
                                last_ik_q[h] = list(q0)
                                cmd_q[h] = list(q0)
                        _say("Recalibrated. Ready")
                        print(f"  {GREEN}Recalibrated. Ready.{RESET}")
                    except Exception as e:
                        print(f"  {RED}Recalibration failed: {e}{RESET}")

            # ── Rate limit (~40 Hz) ─────────────────────────────────────────
            elapsed = time.time() - t_start
            sleep = 0.025 - elapsed
            if sleep > 0:
                time.sleep(sleep)

    except KeyboardInterrupt:
        print("\n\n  Stopped.")

    finally:
        # Restore terminal settings
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, _old_term)

        # Discard any in-progress recording on exit
        if rec_enabled and rec_buf and len(rec_buf) > 0:
            print(f"  {RED}Discarded in-progress recording "
                  f"({len(rec_buf)} frames){RESET}")

        if rec_cameras:
            rec_cameras.release()

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
