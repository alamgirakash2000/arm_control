#!/usr/bin/env python3
"""
Quest Teleop Receiver -- Dual Piper Hanging Arms (SteamVR / OpenVR)
====================================================================
Streams 6DOF pose from Meta Quest 3 (or any SteamVR controller) via OpenVR,
validates every frame with IK, and commands two hanging Piper robot arms.

Simulation mode  (default) : validates IK, displays deltas -- no robot motion.
Robot mode (--with_robot)  : also moves the physical arms via CAN.

Prerequisites:
    1. SteamVR installed and running on this PC.
    2. Meta Quest 3 connected via Air Link, Quest Link, or Virtual Desktop.
       Both controllers must be on and tracked before launching.
    3. pip install openvr opencv-python

    The headset does NOT need to be worn on your face the whole time.
    Hang it from your neck -- the inside-out cameras will still track the
    controllers held in front of you, and you can see the actual robots.

Usage:
    python3 quest_receiver.py                                     # simulation
    python3 quest_receiver.py --with_robot                        # real robot
    python3 quest_receiver.py --with_robot --left can0 --right can1
    python3 quest_receiver.py --scale 0.7 --speed 20 --smoothing 0.5

Calibration:
    After launch you get a countdown to hold both controllers at your robot
    home position.  All motion is computed as:
        robot_target = robot_home + remap(controller_current - controller_home) * scale
    See AXIS_MAP_L / AXIS_MAP_R below and the README for tuning.

OpenVR coordinate frame:
    X = right,  Y = up,  -Z = forward  (same handedness as Vision Pro).
    The AXIS_MAP defaults match teleop_receiver.py so calibration findings
    transfer directly between the two scripts.
"""

import argparse
import os
import subprocess
import sys
import math
import time
import threading

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from piper_core import (
    ik_solve,
    euler_to_rotation,
    rotation_to_euler,
    forward_kinematics,
    joints_in_limits,
    HOME_POSITION,
)

# Auto-install openvr if missing
try:
    import openvr
except ImportError:
    print("  openvr not found -- installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openvr"])
    import openvr  # type: ignore

try:
    import cv2
    _cv2_ok = True
except ImportError:
    _cv2_ok = False


# ── Default configuration ──────────────────────────────────────────────────────

DEFAULT_SCALE     = 1.0    # 1.0 = 1:1,  0.5 = half range,  2.0 = double
DEFAULT_SPEED     = 20     # Robot speed % -- start low
DEFAULT_SMOOTHING = 0.4    # EMA alpha: 0 = frozen, 1 = raw (no smoothing)
GRIPPER_MAX_MM    = 70.0   # Gripper fully-open in mm
GRIPPER_DEADBAND  = 2.0    # % -- only send gripper cmd when delta > this
DEADBAND_CM       = 1.0    # cm  -- tremor suppression for position
DEADBAND_DEG      = 2.0    # deg -- tremor suppression for rotation
POLL_HZ           = 90     # Hz  -- main loop target rate
DISPLAY_HZ        = 30     # Hz  -- terminal / cv2 refresh rate
CONTROLLER_RESCAN_S = 3.0  # s   -- how often to re-scan for controller IDs
REF_SECS          = 2.5    # s   -- seconds to average for home pose capture

# ── Axis remapping ─────────────────────────────────────────────────────────────
# Format: (OpenVR_axis_index, sign) for each robot axis [x, y, z].
# OpenVR frame:  X=right, Y=up, -Z=forward  (same as Vision Pro world frame).
# Tune: in simulation mode, move one controller axis at a time and watch
# which robot delta changes.  See README for the full calibration procedure.
#
# LEFT arm:  controller forward -> robot +x,  up -> robot +z,  left -> robot -y
AXIS_MAP_L = [(1,  1), (0,  1), (2, -1)]   # Y->rob_x,  X->rob_y,  -Z->rob_z
# RIGHT arm: lateral axis mirrored for symmetric dual-arm layout
AXIS_MAP_R = [(1,  1), (0, -1), (2, -1)]   # Y->rob_x, -X->rob_y,  -Z->rob_z

# ── ANSI helpers ───────────────────────────────────────────────────────────────
GREEN  = "\033[32m"
RED    = "\033[31m"
YELLOW = "\033[33m"
RESET  = "\033[0m"
CLEAR  = "\033[2J\033[H"

# ── Local cv2 window size ──────────────────────────────────────────────────────
OW, OH = 1280, 720

# ── Shared display state ───────────────────────────────────────────────────────
_disp = {
    "phase":     "starting",
    "countdown": 0,
    "frame":     0,
    "scale":     DEFAULT_SCALE,
    "sim":       True,
    "left":  dict(tracking=False, ok=False, msg="", dx=0, dy=0, dz=0,
                  dr=0, dp=0, dyaw=0, gripper=100),
    "right": dict(tracking=False, ok=False, msg="", dx=0, dy=0, dz=0,
                  dr=0, dp=0, dyaw=0, gripper=100),
}


# ── OpenVR helpers ─────────────────────────────────────────────────────────────

def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    return q / n if n > 1e-10 else q


def openvr_pose_to_pos_rotmat(pose_mat) -> tuple:
    """Convert OpenVR 3x4 tracking matrix -> (pos[3], R[3,3])."""
    pos = np.array([pose_mat[0][3], pose_mat[1][3], pose_mat[2][3]], dtype=np.float64)
    R   = np.array([
        [pose_mat[0][0], pose_mat[0][1], pose_mat[0][2]],
        [pose_mat[1][0], pose_mat[1][1], pose_mat[1][2]],
        [pose_mat[2][0], pose_mat[2][1], pose_mat[2][2]],
    ], dtype=np.float64)
    return pos, R


def find_controllers(vr_system) -> tuple:
    """
    Return (left_id, right_id) for tracked controller devices.

    Prefers TrackedControllerRole_LeftHand / RightHand role assignments.
    Falls back to assigning the first two un-roled controllers as left/right
    -- common with Quest via Air Link before roles are negotiated.
    """
    controllers = []
    for idx in range(openvr.k_unMaxTrackedDeviceCount):
        try:
            if vr_system.getTrackedDeviceClass(idx) == openvr.TrackedDeviceClass_Controller:
                controllers.append(idx)
        except Exception:
            continue

    left_id = right_id = None
    for idx in controllers:
        try:
            role = vr_system.getControllerRoleForTrackedDeviceIndex(idx)
        except Exception:
            role = 0
        if role == openvr.TrackedControllerRole_LeftHand:
            left_id = idx
        elif role == openvr.TrackedControllerRole_RightHand:
            right_id = idx

    remaining = [i for i in controllers if i not in {left_id, right_id}]
    if left_id  is None and remaining: left_id  = remaining.pop(0)
    if right_id is None and remaining: right_id = remaining.pop(0)
    return left_id, right_id


def poll_controller(vr_system, controller_id, poses) -> tuple:
    """
    Poll a single controller for position, rotation, and trigger.

    Returns (pos, R, trigger, status) where:
        pos     - np.ndarray [x,y,z] metres, or None
        R       - 3x3 rotation matrix, or None
        trigger - float 0.0-1.0  (rAxis[1] = index trigger on Quest Touch)
        status  - "OK" | "DISCONNECTED" | "NO TRACKING" | "NOT FOUND"
    """
    if controller_id is None:
        return None, None, 0.0, "NOT FOUND"

    pose = poses[controller_id]
    if not pose.bDeviceIsConnected:
        return None, None, 0.0, "DISCONNECTED"
    if not pose.bPoseIsValid:
        return None, None, 0.0, "NO TRACKING"

    pos, R = openvr_pose_to_pos_rotmat(pose.mDeviceToAbsoluteTracking)

    trigger = 0.0
    try:
        success, state = vr_system.getControllerState(controller_id)
        if success and len(state.rAxis) > 1:
            trigger = float(np.clip(state.rAxis[1].x, 0.0, 1.0))
    except Exception:
        pass

    return pos, R, trigger, "OK"


def capture_reference(vr_system, left_id, right_id,
                      secs: float, prediction: float, poll_interval: float) -> dict:
    """
    Average controller poses over `secs` seconds of valid dual-hand tracking.
    Active time only counts when both controllers report bPoseIsValid.

    Returns {"left": (pos, R), "right": (pos, R)}.
    """
    pos_samples = {"left": [], "right": []}
    R_samples   = {"left": [], "right": []}
    active_t    = 0.0
    t_wall0     = time.time()
    last_len    = 0

    while active_t < secs:
        if time.time() - t_wall0 > 120.0:
            raise RuntimeError(
                "Controllers not tracked for 2 minutes during reference capture.\n"
                "Check SteamVR tracking and headset wake state."
            )

        poses = vr_system.getDeviceToAbsoluteTrackingPose(
            openvr.TrackingUniverseStanding, prediction, openvr.k_unMaxTrackedDeviceCount
        )
        ok = {}
        for cid, hand in ((left_id, "left"), (right_id, "right")):
            pos, R, _, status = poll_controller(vr_system, cid, poses)
            ok[hand] = status == "OK"
            if ok[hand]:
                pos_samples[hand].append(pos)
                R_samples[hand].append(R)

        if ok["left"] and ok["right"]:
            active_t += poll_interval

        rem  = max(0.0, secs - active_t)
        line = (f"  ref remaining: {rem:4.1f}s  "
                f"[L:{'OK' if ok['left'] else 'wait'} "
                f"R:{'OK' if ok['right'] else 'wait'}]")
        pad  = max(0, last_len - len(line))
        sys.stdout.write("\r" + line + " " * pad)
        sys.stdout.flush()
        last_len = len(line)
        time.sleep(max(0.0, poll_interval))

    sys.stdout.write("\n")

    out = {}
    for hand in ("left", "right"):
        n = len(pos_samples[hand])
        if n < 5:
            raise RuntimeError(f"Too few reference samples for {hand.upper()} (n={n}).")
        pos_mean = np.mean(np.stack(pos_samples[hand]), axis=0)
        # SVD-based mean rotation (proper SO(3) average)
        R_sum = np.sum(np.stack(R_samples[hand]), axis=0)
        U, _, Vt = np.linalg.svd(R_sum)
        R_mean = U @ Vt
        if np.linalg.det(R_mean) < 0:
            U[:, -1] *= -1
            R_mean = U @ Vt
        out[hand] = (pos_mean, R_mean)

    return out


# ── Non-blocking robot commander ───────────────────────────────────────────────

class ArmCommander:
    """
    Background-thread Cartesian commander.

    The 90 Hz main loop calls .send() freely; stale targets are overwritten
    so only the freshest one is executed.  Both arms run in separate threads,
    so left and right commands are truly parallel.

    Effective robot command rate: ~1 / CMD_TIMEOUT_S = 20 Hz.
    The robot firmware interpolates between commands, giving smooth motion.
    """

    CMD_TIMEOUT_S = 0.05    # s -- max wait per send_cartesian (~20 Hz cmd rate)

    def __init__(self, ctrl, label="arm"):
        self._ctrl    = ctrl
        self._target  = None
        self._grip    = None
        self._running = True
        self._cond    = threading.Condition(threading.Lock())
        threading.Thread(target=self._run, daemon=True, name=f"arm-{label}").start()

    def send(self, x, y, z, roll, pitch, yaw):
        with self._cond:
            self._target = (x, y, z, roll, pitch, yaw)
            self._cond.notify()

    def gripper(self, mm: float):
        with self._cond:
            self._grip = float(mm)
            self._cond.notify()

    def stop(self):
        with self._cond:
            self._running = False
            self._cond.notify()

    def _run(self):
        while True:
            with self._cond:
                while self._running and self._target is None and self._grip is None:
                    self._cond.wait(timeout=0.05)
                target = self._target
                grip   = self._grip
                self._target = None
                self._grip   = None
                if not self._running:
                    break
            if target is not None:
                try:
                    self._ctrl.send_cartesian(*target, timeout=self.CMD_TIMEOUT_S)
                except Exception:
                    pass
            if grip is not None:
                try:
                    self._ctrl.gripper_ctrl(grip)
                except Exception:
                    pass


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Meta Quest 3 / SteamVR teleop for dual Piper hanging arms.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 quest_receiver.py                                    # simulation
  python3 quest_receiver.py --with_robot                       # real robot
  python3 quest_receiver.py --with_robot --left can0 --right can1
  python3 quest_receiver.py --scale 0.5 --speed 15 --smoothing 0.6
        """,
    )
    p.add_argument("--with_robot",     action="store_true",
                   help="Enable real CAN commands (default: simulation only)")
    p.add_argument("--left",           default=None, metavar="IFACE",
                   help="CAN interface for left arm  (default: auto-detect)")
    p.add_argument("--right",          default=None, metavar="IFACE",
                   help="CAN interface for right arm (default: auto-detect)")
    p.add_argument("--scale",          default=DEFAULT_SCALE, type=float,
                   help=f"Motion scale 0.1-2.0 (default: {DEFAULT_SCALE})")
    p.add_argument("--speed",          default=DEFAULT_SPEED, type=int,
                   help=f"Robot speed %% 1-100 (default: {DEFAULT_SPEED})")
    p.add_argument("--smoothing",      default=DEFAULT_SMOOTHING, type=float,
                   help=f"EMA alpha 0-1: 0=frozen 1=raw (default: {DEFAULT_SMOOTHING})")
    p.add_argument("--prediction",     default=0.010, type=float,
                   help="OpenVR pose prediction horizon seconds (default: 0.010)")
    p.add_argument("--ref-secs",       default=REF_SECS, type=float,
                   help=f"Seconds to average for home pose capture (default: {REF_SECS})")
    p.add_argument("--invert-gripper", action="store_true",
                   help="Invert trigger: fully pressed = open instead of close")
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


# ── Math helpers ───────────────────────────────────────────────────────────────

def rotation_delta(R_cur: np.ndarray, R_home: np.ndarray) -> np.ndarray:
    return R_cur @ R_home.T


def deadband(v: float, threshold: float) -> float:
    return 0.0 if abs(v) < threshold else v


def remap(pos_delta_m: np.ndarray, axis_map: list) -> np.ndarray:
    return np.array([pos_delta_m[ax] * sign for ax, sign in axis_map])


# ── IK validation ──────────────────────────────────────────────────────────────

def validate(dx_m, dy_m, dz_m, dr_rad, dp_rad, dyaw_rad,
             warmstart_q, home_pos, home_rpy):
    target_pos = [home_pos[0] + dx_m, home_pos[1] + dy_m, home_pos[2] + dz_m]
    target_rpy = [home_rpy[0] + dr_rad, home_rpy[1] + dp_rad, home_rpy[2] + dyaw_rad]
    target_rot = euler_to_rotation(*target_rpy)

    q_sol = ik_solve(warmstart_q, target_pos, target_rot)
    if q_sol is None:
        return False, "IK failed - unreachable", None
    if not joints_in_limits(q_sol):
        return False, "joint limits exceeded", None

    T_check, _ = forward_kinematics(q_sol)
    pos_err = np.linalg.norm(np.array(target_pos) - T_check[:3, 3])
    if pos_err > 0.01:
        return False, f"IK diverged ({pos_err * 100:.1f} cm err)", None

    return True, "OK", q_sol


# ── Local cv2 overlay ──────────────────────────────────────────────────────────

def _txt(img, text, x, y, scale, color, thick=2):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thick, cv2.LINE_AA)


def _bar(img, x, y, w, h, pct, color):
    cv2.rectangle(img, (x, y), (x + w, y + h), (60, 60, 60), -1)
    fill = int(w * max(0.0, min(100.0, pct)) / 100)
    if fill > 0:
        cv2.rectangle(img, (x, y), (x + fill, y + h), color, -1)


def render_frame() -> np.ndarray:
    img   = np.full((OH, OW, 3), 18, dtype=np.uint8)
    d     = _disp
    phase = d["phase"]

    if phase == "starting":
        _txt(img, "Starting SteamVR...", 300, 360, 2.0, (200, 200, 200), 3)

    elif phase == "waiting":
        _txt(img, "Waiting for both controllers...", 80, 360, 1.7, (200, 200, 0), 3)

    elif phase == "calibrating":
        _txt(img, "HOLD HOME POSITION",       160, 300, 2.2, (0, 220, 220), 4)
        _txt(img, f"Snapping home in  {d['countdown']} s",
             300, 420, 1.6, (255, 255, 255), 2)

    else:
        mode_col = (0, 180, 0) if not d["sim"] else (180, 140, 0)
        mode_txt = "ROBOT" if not d["sim"] else "SIMULATION"
        _txt(img,
             f"PIPER QUEST TELEOP  [{mode_txt}]  frame {d['frame']}  scale {d['scale']:.2f}",
             20, 42, 0.85, mode_col, 2)
        cv2.line(img, (0, 55), (OW, 55), (60, 60, 60), 1)

        for i, side in enumerate(("left", "right")):
            s     = d[side]
            label = "LEFT" if side == "left" else "RIGHT"
            bx    = 20 + i * 640
            by    = 80

            trk_col = (0, 200, 0) if s["tracking"] else (0, 0, 200)
            trk_txt = "tracking"  if s["tracking"] else "LOST"
            ik_col  = (0, 220, 0) if s["ok"]       else (60, 80, 220)
            ik_txt  = "IK OK"     if s["ok"]       else f"!! {s['msg']}"

            _txt(img, label,   bx,       by + 36, 1.6, (230, 230, 230), 3)
            _txt(img, trk_txt, bx + 140, by + 36, 1.0, trk_col, 2)

            by += 60
            _txt(img, ik_txt[:36], bx, by, 0.85, ik_col, 2)

            by += 44
            _txt(img, f"dx {s['dx']:+6.2f}cm", bx,       by, 0.85, (200, 200, 200), 2)
            _txt(img, f"dy {s['dy']:+6.2f}cm", bx + 180, by, 0.85, (200, 200, 200), 2)
            _txt(img, f"dz {s['dz']:+6.2f}cm", bx + 360, by, 0.85, (200, 200, 200), 2)

            by += 40
            _txt(img, f"dr  {s['dr']:+6.1f} deg",   bx,       by, 0.85, (200, 200, 200), 2)
            _txt(img, f"dp  {s['dp']:+6.1f} deg",   bx + 180, by, 0.85, (200, 200, 200), 2)
            _txt(img, f"yaw {s['dyaw']:+6.1f} deg", bx + 360, by, 0.85, (200, 200, 200), 2)

            by += 44
            _txt(img, f"gripper {s['gripper']:5.1f}%  [trigger]",
                 bx, by, 0.85, (200, 200, 200), 2)
            _bar(img, bx + 310, by - 22, 280, 26, s["gripper"], ik_col)

            if i == 0:
                cv2.line(img, (638, 60), (638, OH - 20), (60, 60, 60), 1)

        footer = "Ctrl+C or ESC to stop  |  Restart to re-calibrate"
        if d["sim"]:
            footer += "   [SIMULATION - no robot motion]"
        _txt(img, footer, 20, OH - 20, 0.6, (80, 80, 80), 1)

    return img


# ── Terminal helpers ───────────────────────────────────────────────────────────

def print_arm(label, s):
    color     = GREEN if s["ok"] else RED
    tick      = "OK" if s["ok"] else "!!"
    track_dot = f"{GREEN}[on]{RESET}" if s["tracking"] else f"{RED}[lost]{RESET}"
    print(f"  {label}  {track_dot}   IK: {color}{tick} {s['msg']}{RESET}")
    print(f"    dx {s['dx']:+7.2f} cm    dy {s['dy']:+7.2f} cm    dz {s['dz']:+7.2f} cm")
    print(f"    dr {s['dr']:+7.2f} deg   dp {s['dp']:+7.2f} deg   yaw {s['dyaw']:+7.2f} deg")
    print(f"    gripper  {s['gripper']:5.1f} %  [index trigger]")


def countdown_display(seconds: int, label: str):
    for i in range(seconds, 0, -1):
        _disp["countdown"] = i
        t_end = time.time() + 1.0
        while time.time() < t_end:
            if _cv2_ok:
                cv2.imshow("Piper Quest Teleop", render_frame())
                cv2.waitKey(1)
            print(f"\r  {label}  {i:2d} s ...  ", end="", flush=True)
            time.sleep(1 / 30.0)
    print()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args        = parse_args()
    scale       = max(0.1, min(2.0, args.scale))
    with_robot  = args.with_robot
    alpha       = float(np.clip(args.smoothing, 0.0, 1.0))
    prediction  = float(max(0.0, args.prediction))
    invert_grip = bool(args.invert_gripper)

    _disp["scale"] = scale
    _disp["sim"]   = not with_robot

    print()
    print("=" * 62)
    print("  QUEST TELEOP - DUAL PIPER HANGING ARMS  (SteamVR / OpenVR)")
    print("=" * 62)
    print(f"  Mode      : {'ROBOT (CAN active)' if with_robot else 'SIMULATION  (no robot motion)'}")
    print(f"  Scale     : {scale}    Deadband: {DEADBAND_CM} cm / {DEADBAND_DEG} deg")
    print(f"  Smoothing : {alpha}  (EMA -- 0=frozen, 1=raw)")
    if with_robot:
        print(f"  Speed     : {args.speed} %")
    print(f"  Overlay   : {'cv2 local window' if _cv2_ok else 'terminal only'}")
    print()
    print("  Ensure SteamVR is running and Quest is connected (Air Link / Quest Link).")
    print()

    # ── CAN port resolution ──────────────────────────────────────────────────
    left_can = right_can = None
    if with_robot:
        left_can  = args.left
        right_can = args.right
        if left_can is None or right_can is None:
            auto_l, auto_r = get_can_ports()
            if left_can  is None: left_can  = auto_l
            if right_can is None: right_can = auto_r
        if not left_can or not right_can:
            print("  ERROR: Cannot determine CAN ports.")
            print("  Run:  sudo bash start_teleop.sh")
            print("  Or:   --with_robot --left can0 --right can1")
            sys.exit(1)
        print(f"  CAN  : LEFT={left_can}   RIGHT={right_can}")
        print()

    # ── Precompute robot home reference ───────────────────────────────────────
    T_home, _      = forward_kinematics(HOME_POSITION)
    home_robot_pos = list(T_home[:3, 3])
    home_robot_rpy = list(rotation_to_euler(T_home[:3, :3]))

    # ── SteamVR init ─────────────────────────────────────────────────────────
    _disp["phase"] = "starting"
    if _cv2_ok:
        cv2.imshow("Piper Quest Teleop", render_frame())
        cv2.waitKey(1)

    print("  Initialising SteamVR (OpenVR) ...", end=" ", flush=True)
    try:
        vr_system = openvr.init(openvr.VRApplication_Other)
    except openvr.OpenVRError as e:
        print(f"FAILED\n  {e}")
        print("  Make sure SteamVR is running and the headset is awake.")
        sys.exit(1)
    print("OK")

    # ── Wait for both controllers ─────────────────────────────────────────────
    _disp["phase"] = "waiting"
    print("  Waiting for both controllers to be tracked ...", end=" ", flush=True)

    left_id = right_id = None
    last_rescan_t = 0.0
    t_wait0 = time.time()

    while True:
        now = time.time()
        if now - last_rescan_t >= 1.0:
            left_id, right_id = find_controllers(vr_system)
            last_rescan_t = now

        poses = vr_system.getDeviceToAbsoluteTrackingPose(
            openvr.TrackingUniverseStanding, prediction, openvr.k_unMaxTrackedDeviceCount
        )
        _, _, _, ls = poll_controller(vr_system, left_id,  poses)
        _, _, _, rs = poll_controller(vr_system, right_id, poses)
        if ls == "OK" and rs == "OK":
            break

        if now - t_wait0 > 120.0:
            print("\n  TIMEOUT: controllers not tracked within 2 minutes.")
            openvr.shutdown()
            sys.exit(1)

        if _cv2_ok:
            cv2.imshow("Piper Quest Teleop", render_frame())
            cv2.waitKey(1)
        time.sleep(0.1)

    print("OK")
    print(f"  Controllers found: LEFT={left_id}  RIGHT={right_id}")
    print()

    # ── Capture home reference pose ───────────────────────────────────────────
    _disp["phase"] = "calibrating"
    print(f"  {YELLOW}Hold BOTH controllers at your robot home position.{RESET}")
    print(f"  {YELLOW}Keep them perfectly still -- capturing controller home pose ...{RESET}")
    countdown_display(int(args.ref_secs) + 3, "Snapping home in")

    refs = capture_reference(
        vr_system, left_id, right_id,
        secs=float(args.ref_secs),
        prediction=prediction,
        poll_interval=1.0 / POLL_HZ,
    )
    home_pos_L, home_R_L = refs["left"]    # np.ndarray [3],  np.ndarray [3,3]
    home_pos_R, home_R_R = refs["right"]

    _disp["phase"] = "streaming"
    print(f"  {GREEN}Home pose captured -- teleoperation active.{RESET}")
    print()

    # ── Connect robot arms ────────────────────────────────────────────────────
    left_ctrl = right_ctrl = None
    left_cmd  = right_cmd  = None

    if with_robot:
        from piper_core import PiperHangingController

        left_ctrl  = PiperHangingController(can_port=left_can)
        right_ctrl = PiperHangingController(can_port=right_can)

        print("  Connecting arms ...")
        results = [False, False]
        def _cl(): results[0] = left_ctrl.connect()
        def _cr(): results[1] = right_ctrl.connect()
        t1 = threading.Thread(target=_cl)
        t2 = threading.Thread(target=_cr)
        t1.start(); t2.start(); t1.join(); t2.join()

        if not results[0]:
            print("  LEFT arm failed -- aborting.")
            openvr.shutdown(); sys.exit(1)
        if not results[1]:
            print("  RIGHT arm failed -- aborting.")
            left_ctrl.shutdown(); openvr.shutdown(); sys.exit(1)

        left_ctrl.speed  = args.speed
        right_ctrl.speed = args.speed

        print("  Homing both arms ...")
        t1 = threading.Thread(target=left_ctrl.go_home)
        t2 = threading.Thread(target=right_ctrl.go_home)
        t1.start(); t2.start(); t1.join(); t2.join()
        print("  Arms ready.")
        print()

        left_cmd  = ArmCommander(left_ctrl,  "left")
        right_cmd = ArmCommander(right_ctrl, "right")

    # ── State: IK warm-start + EMA smoothed deltas ────────────────────────────
    left_joints  = list(HOME_POSITION)
    right_joints = list(HOME_POSITION)

    # EMA smoothed delta state (cm / deg) -- persists across tracking loss
    sdx_L = sdy_L = sdz_L = sdr_L = sdp_L = sdyaw_L = 0.0
    sdx_R = sdy_R = sdz_R = sdr_R = sdp_R = sdyaw_R = 0.0

    last_grip_L  = 100.0
    last_grip_R  = 100.0
    status_L     = "INIT"
    status_R     = "INIT"
    frame        = 0
    last_display = 0.0
    last_rescan  = time.time()

    sim_note = f"  {YELLOW}[SIMULATION - no robot motion]{RESET}" if not with_robot else ""
    print(f"  Streaming -- Ctrl+C or ESC to stop  |  Restart to re-calibrate{sim_note}")
    print()

    # ── Main control loop ─────────────────────────────────────────────────────
    try:
        while True:
            t_start = time.time()

            # Periodically re-scan controller IDs (handles reconnects / role negotiation)
            if t_start - last_rescan >= CONTROLLER_RESCAN_S:
                new_l, new_r = find_controllers(vr_system)
                if (new_l, new_r) != (left_id, right_id):
                    left_id, right_id = new_l, new_r
                    print(f"\n  [INFO] Controllers refreshed: L={left_id} R={right_id}")
                last_rescan = t_start

            poses = vr_system.getDeviceToAbsoluteTrackingPose(
                openvr.TrackingUniverseStanding, prediction, openvr.k_unMaxTrackedDeviceCount
            )

            # ── LEFT controller ───────────────────────────────────────────────
            pos_L, R_L, trig_L, new_sl = poll_controller(vr_system, left_id, poses)
            left_ok = new_sl == "OK"
            if new_sl != status_L:
                print(f"\n  [LEFT] {new_sl}")
                status_L = new_sl

            if left_ok:
                # Position: remap → cm → deadband → EMA smooth
                raw_dp_L = remap((pos_L - home_pos_L) * scale, AXIS_MAP_L)
                sdx_L = alpha * deadband(raw_dp_L[0] * 100, DEADBAND_CM) + (1 - alpha) * sdx_L
                sdy_L = alpha * deadband(raw_dp_L[1] * 100, DEADBAND_CM) + (1 - alpha) * sdy_L
                sdz_L = alpha * deadband(raw_dp_L[2] * 100, DEADBAND_CM) + (1 - alpha) * sdz_L

                # Rotation: delta matrix → euler → deadband → EMA smooth
                rL, pL, yL = rotation_to_euler(rotation_delta(R_L, home_R_L))
                sdr_L   = alpha * deadband(math.degrees(rL), DEADBAND_DEG) + (1 - alpha) * sdr_L
                sdp_L   = alpha * deadband(math.degrees(pL), DEADBAND_DEG) + (1 - alpha) * sdp_L
                sdyaw_L = alpha * deadband(math.degrees(yL), DEADBAND_DEG) + (1 - alpha) * sdyaw_L

                # Trigger → gripper %  (fully pressed = close by default)
                t_val_L   = 1.0 - trig_L if invert_grip else trig_L
                gripper_L = float(np.clip(t_val_L * 100.0, 0.0, 100.0))
            else:
                gripper_L = last_grip_L   # hold on tracking loss

            dx_L, dy_L, dz_L       = sdx_L, sdy_L, sdz_L
            dr_L, dp_L, dyaw_L     = sdr_L, sdp_L, sdyaw_L

            # ── RIGHT controller ──────────────────────────────────────────────
            pos_R, R_R, trig_R, new_sr = poll_controller(vr_system, right_id, poses)
            right_ok = new_sr == "OK"
            if new_sr != status_R:
                print(f"\n  [RIGHT] {new_sr}")
                status_R = new_sr

            if right_ok:
                raw_dp_R = remap((pos_R - home_pos_R) * scale, AXIS_MAP_R)
                sdx_R = alpha * deadband(raw_dp_R[0] * 100, DEADBAND_CM) + (1 - alpha) * sdx_R
                sdy_R = alpha * deadband(raw_dp_R[1] * 100, DEADBAND_CM) + (1 - alpha) * sdy_R
                sdz_R = alpha * deadband(raw_dp_R[2] * 100, DEADBAND_CM) + (1 - alpha) * sdz_R

                rR, pR, yR = rotation_to_euler(rotation_delta(R_R, home_R_R))
                sdr_R   = alpha * deadband(math.degrees(rR), DEADBAND_DEG) + (1 - alpha) * sdr_R
                sdp_R   = alpha * deadband(math.degrees(pR), DEADBAND_DEG) + (1 - alpha) * sdp_R
                sdyaw_R = alpha * deadband(math.degrees(yR), DEADBAND_DEG) + (1 - alpha) * sdyaw_R

                t_val_R   = 1.0 - trig_R if invert_grip else trig_R
                gripper_R = float(np.clip(t_val_R * 100.0, 0.0, 100.0))
            else:
                gripper_R = last_grip_R

            dx_R, dy_R, dz_R   = sdx_R, sdy_R, sdz_R
            dr_R, dp_R, dyaw_R = sdr_R, sdp_R, sdyaw_R

            # ── IK validation ─────────────────────────────────────────────────
            lok, lmsg, lq = validate(
                dx_L / 100, dy_L / 100, dz_L / 100,
                math.radians(dr_L), math.radians(dp_L), math.radians(dyaw_L),
                left_joints, home_robot_pos, home_robot_rpy,
            )
            rok, rmsg, rq = validate(
                dx_R / 100, dy_R / 100, dz_R / 100,
                math.radians(dr_R), math.radians(dp_R), math.radians(dyaw_R),
                right_joints, home_robot_pos, home_robot_rpy,
            )
            if lok and lq is not None: left_joints  = lq
            if rok and rq is not None: right_joints = rq

            # ── Robot commands -- non-blocking, parallel ───────────────────────
            if with_robot:
                if left_cmd is not None and lok:
                    left_cmd.send(
                        home_robot_pos[0] + dx_L / 100,
                        home_robot_pos[1] + dy_L / 100,
                        home_robot_pos[2] + dz_L / 100,
                        home_robot_rpy[0] + math.radians(dr_L),
                        home_robot_rpy[1] + math.radians(dp_L),
                        home_robot_rpy[2] + math.radians(dyaw_L),
                    )
                if right_cmd is not None and rok:
                    right_cmd.send(
                        home_robot_pos[0] + dx_R / 100,
                        home_robot_pos[1] + dy_R / 100,
                        home_robot_pos[2] + dz_R / 100,
                        home_robot_rpy[0] + math.radians(dr_R),
                        home_robot_rpy[1] + math.radians(dp_R),
                        home_robot_rpy[2] + math.radians(dyaw_R),
                    )
                # Gripper -- deadband-gated to avoid CAN spam
                if left_cmd is not None and abs(gripper_L - last_grip_L) >= GRIPPER_DEADBAND:
                    left_cmd.gripper(gripper_L / 100.0 * GRIPPER_MAX_MM)
                    last_grip_L = gripper_L
                if right_cmd is not None and abs(gripper_R - last_grip_R) >= GRIPPER_DEADBAND:
                    right_cmd.gripper(gripper_R / 100.0 * GRIPPER_MAX_MM)
                    last_grip_R = gripper_R

            frame += 1

            # ── Display -- throttled to DISPLAY_HZ ────────────────────────────
            now = time.time()
            if now - last_display >= 1.0 / DISPLAY_HZ:
                last_display = now
                _disp["frame"] = frame
                _disp["left"]  = dict(
                    tracking=left_ok,  ok=lok, msg=lmsg,
                    dx=dx_L, dy=dy_L, dz=dz_L,
                    dr=dr_L, dp=dp_L, dyaw=dyaw_L, gripper=gripper_L,
                )
                _disp["right"] = dict(
                    tracking=right_ok, ok=rok, msg=rmsg,
                    dx=dx_R, dy=dy_R, dz=dz_R,
                    dr=dr_R, dp=dp_R, dyaw=dyaw_R, gripper=gripper_R,
                )

                if _cv2_ok:
                    cv2.imshow("Piper Quest Teleop", render_frame())
                    if cv2.waitKey(1) == 27:    # ESC to stop
                        break

                sim_label = f"   {YELLOW}[SIM]{RESET}" if not with_robot else ""
                print(CLEAR, end="")
                print(f"  QUEST TELEOP   frame={frame}   scale={scale:.2f}   "
                      f"deadband={DEADBAND_CM}cm   smooth={alpha:.2f}{sim_label}")
                print()
                print_arm("LEFT ", _disp["left"])
                print()
                print_arm("RIGHT", _disp["right"])
                print()
                print("  Ctrl+C or ESC to stop  |  Restart to re-calibrate home")

            # ── Rate limit ────────────────────────────────────────────────────
            elapsed = time.time() - t_start
            sleep   = 1.0 / POLL_HZ - elapsed
            if sleep > 0:
                time.sleep(sleep)

    except KeyboardInterrupt:
        print("\n\n  Stopped.")

    finally:
        if left_cmd  is not None: left_cmd.stop()
        if right_cmd is not None: right_cmd.stop()

        if with_robot and (left_ctrl is not None or right_ctrl is not None):
            print("  Relaxing arms ...", end=" ", flush=True)
            try:
                threads = []
                if left_ctrl  is not None:
                    threads.append(threading.Thread(target=left_ctrl.go_relax))
                if right_ctrl is not None:
                    threads.append(threading.Thread(target=right_ctrl.go_relax))
                for t in threads: t.start()
                for t in threads: t.join()
                print("done")
            except Exception as e:
                print(f"skipped ({e})")

            print("  Shutting down arms ...", end=" ", flush=True)
            try:
                if left_ctrl  is not None: left_ctrl.shutdown()
                if right_ctrl is not None: right_ctrl.shutdown()
                print("done")
            except Exception as e:
                print(f"skipped ({e})")

        try:
            openvr.shutdown()
            print("  SteamVR disconnected.")
        except Exception:
            pass

        if _cv2_ok:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
