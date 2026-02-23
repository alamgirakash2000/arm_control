#!/usr/bin/env python3
"""
Vision Pro Teleop Receiver — Dual Piper Hanging Arms
=====================================================
Streams wrist pose from Apple Vision Pro, validates every frame with IK,
and commands two hanging Piper robot arms in real time.

Simulation mode  (default) : validates IK, displays deltas — no robot motion.
Robot mode (--with_robot)  : also moves the physical arms via CAN.

Requirements:
    pip install avp_stream opencv-python

Usage:
    python3 teleop_receiver.py                                    # simulation
    python3 teleop_receiver.py --with_robot                       # real robot (auto-detect CAN)
    python3 teleop_receiver.py --with_robot --left can0 --right can1
    python3 teleop_receiver.py --ip 10.0.0.143 --scale 0.7 --speed 25

Calibration:
    On startup you get a 10-second countdown to hold both hands at your
    robot's home position.  From that snapshot, all motion is computed as
    (hand_current - hand_home) → robot EE delta relative to its own home.

Axis remapping:
    AXIS_MAP_L / AXIS_MAP_R at the top of this file map Vision Pro axes to
    robot EE axes.  See the README for the calibration procedure.
"""

import argparse
import sys
import os
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

try:
    from avp_stream import VisionProStreamer
except ImportError:
    print("  ERROR: avp_stream not installed.")
    print("  Run:   pip install avp_stream")
    sys.exit(1)

try:
    import cv2
    _cv2_ok = True
except ImportError:
    _cv2_ok = False

# ── Default configuration (override via CLI) ───────────────────────────────────

#DEFAULT_IP       = "10.0.0.143"   # Vision Pro IP — check Settings → Wi-Fi
DEFAULT_IP       = "10.130.17.123"   # Vision Pro IP — check Settings → Wi-Fi
DEFAULT_SCALE    = 1.0            # 1.0 = 1:1,  0.5 = half range,  2.0 = double
DEFAULT_ROT_SCALE = 0.0           # 0.0 = position only, 0.1 = gentle, 1.0 = full rotation
DEFAULT_SPEED    = 50             # Robot speed % — start low, raise once OK
GRIPPER_MAX_MM   = 70.0           # Gripper fully-open distance in mm
GRIPPER_DEADBAND = 2.0            # % — only send gripper cmd when Δ > this
DEADBAND_CM      = 1.0            # cm  — tremor suppression for position
DEADBAND_DEG     = 2.0            # deg — tremor suppression for rotation
PINCH_MAX_M      = 0.08           # m   — pinch distance that maps to 100% open
POLL_HZ          = 10             # Hz  — Vision Pro polling rate
DISPLAY_HZ       = 30             # Hz  — terminal / overlay refresh rate

# ── Axis remapping ─────────────────────────────────────────────────────────────
# Each entry is (VP_axis_index, sign) for robot axes [x, y, z].
# The Vision Pro wrist matrix columns are x=right, y=up, z=back (right-hand).
# Tune by moving one hand axis at a time and watching which robot axis responds.
#
# LEFT arm:  hand forward → robot +x,  hand up → robot +z,  hand left → robot -y
AXIS_MAP_L = [(1,  1), (0,  1), (2, -1)]   # VP_y → rob_x,  VP_x → rob_y,  -VP_z → rob_z
# RIGHT arm: lateral axis mirrored for symmetric dual-arm layout
AXIS_MAP_R = [(1,  1), (0, -1), (2, -1)]   # VP_y → rob_x, -VP_x → rob_y,  -VP_z → rob_z

# ── ANSI terminal helpers ──────────────────────────────────────────────────────
GREEN  = "\033[32m"
RED    = "\033[31m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
RESET  = "\033[0m"
CLEAR  = "\033[2J\033[H"

# ── OpenCV overlay frame size ──────────────────────────────────────────────────
OW, OH = 1280, 720

# ── Shared display state (written by main loop, read by render) ────────────────
_disp = {
    "phase":     "starting",   # starting | waiting | calibrating | streaming
    "countdown": 0,
    "frame":     0,
    "scale":     DEFAULT_SCALE,
    "sim":       True,         # False when --with_robot
    "left":  dict(tracking=False, ok=False, msg="", dx=0, dy=0, dz=0,
                  dr=0, dp=0, dyaw=0, gripper=100),
    "right": dict(tracking=False, ok=False, msg="", dx=0, dy=0, dz=0,
                  dr=0, dp=0, dyaw=0, gripper=100),
}


# ── Non-blocking robot commander ───────────────────────────────────────────────

class ArmCommander:
    """
    Sends send_cartesian / gripper_ctrl commands to a PiperHangingController
    in a background daemon thread so the 90 Hz polling loop is never stalled.

    The main loop calls .send() and .gripper() freely; stale targets are
    silently overwritten — only the freshest target is ever executed.

    Robot command rate is determined by how fast send_cartesian returns
    (CMD_TIMEOUT_S).  50 ms → ~20 Hz effective command rate, which is
    plenty for smooth robot tracking with firmware-side interpolation.
    """

    CMD_TIMEOUT_S = 0.05      # s — max wait per send_cartesian call

    def __init__(self, ctrl, label="arm"):
        self._ctrl    = ctrl
        self._label   = label
        self._target  = None       # (x, y, z, roll, pitch, yaw) or None
        self._grip    = None       # mm float or None
        self._running = True
        self._cond    = threading.Condition(threading.Lock())
        t = threading.Thread(target=self._run, daemon=True, name=f"arm-{label}")
        t.start()

    # ── Public API ─────────────────────────────────────────────────────────────

    def send(self, x, y, z, roll, pitch, yaw):
        """Set the latest Cartesian target (overwrites any pending target)."""
        with self._cond:
            self._target = (x, y, z, roll, pitch, yaw)
            self._cond.notify()

    def gripper(self, mm):
        """Set gripper opening in mm."""
        with self._cond:
            self._grip = float(mm)
            self._cond.notify()

    def stop(self):
        """Signal the background thread to exit."""
        with self._cond:
            self._running = False
            self._cond.notify()

    # ── Background worker ──────────────────────────────────────────────────────

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

            # Call send_cartesian directly to avoid move_to's print spam
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


# ── CLI argument parsing ───────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Vision Pro teleop for dual Piper hanging arms.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 teleop_receiver.py                                   # simulation (safe, no robot motion)
  python3 teleop_receiver.py --with_robot                      # real robot (auto-detect CAN ports)
  python3 teleop_receiver.py --with_robot --left can0 --right can1
  python3 teleop_receiver.py --ip 10.0.0.1 --scale 0.5 --speed 15
        """,
    )
    p.add_argument(
        "--with_robot", action="store_true",
        help="Enable real robot commands via CAN  (default: simulation only)",
    )
    p.add_argument(
        "--left", default=None, metavar="IFACE",
        help="CAN interface for the LEFT arm  (default: auto-detect from start_teleop.sh)",
    )
    p.add_argument(
        "--right", default=None, metavar="IFACE",
        help="CAN interface for the RIGHT arm (default: auto-detect from start_teleop.sh)",
    )
    p.add_argument(
        "--ip", default=DEFAULT_IP, metavar="ADDR",
        help=f"Vision Pro IP address  (default: {DEFAULT_IP})",
    )
    p.add_argument(
        "--scale", default=DEFAULT_SCALE, type=float,
        help=f"Position scale 0.1-2.0  (default: {DEFAULT_SCALE})",
    )
    p.add_argument(
        "--rot_scale", default=DEFAULT_ROT_SCALE, type=float,
        help=f"Rotation scale 0.0-1.0  (0=position only, default: {DEFAULT_ROT_SCALE})",
    )
    p.add_argument(
        "--speed", default=DEFAULT_SPEED, type=int,
        help=f"Robot speed %% 1-100  (default: {DEFAULT_SPEED})",
    )
    return p.parse_args()


def get_can_ports():
    """Read LEFT/RIGHT CAN assignments written by start_teleop.sh."""
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

def rotation_delta(R_cur, R_home):
    """Rotation matrix from home to current: R_delta = R_cur @ R_home.T"""
    return R_cur @ R_home.T


def deadband(v, threshold):
    """Zero out values below threshold (tremor suppression)."""
    return 0.0 if abs(v) < threshold else v


def remap(pos_delta_m, axis_map):
    """Permute and sign-flip a 3-vector according to axis_map."""
    return np.array([pos_delta_m[ax] * sign for ax, sign in axis_map])


# ── IK validation ──────────────────────────────────────────────────────────────

def validate(dx_m, dy_m, dz_m, dr_rad, dp_rad, dyaw_rad,
             warmstart_q, home_pos, home_rpy):
    """
    Check whether (robot_home + delta) is reachable via IK.

    dx/dy/dz are in metres; dr/dp/dyaw are in radians.
    warmstart_q is the last accepted IK solution (speeds convergence).
    home_pos / home_rpy are the robot home EE pose from forward_kinematics.

    Returns (ok: bool, message: str, q_solution or None).
    """
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


# ── OpenCV overlay renderer ────────────────────────────────────────────────────

def _txt(img, text, x, y, scale, color, thick=2):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thick, cv2.LINE_AA)


def _bar(img, x, y, w, h, pct, color):
    cv2.rectangle(img, (x, y), (x + w, y + h), (60, 60, 60), -1)
    fill = int(w * max(0.0, min(100.0, pct)) / 100)
    if fill > 0:
        cv2.rectangle(img, (x, y), (x + fill, y + h), color, -1)


def render_frame():
    img   = np.full((OH, OW, 3), 18, dtype=np.uint8)   # dark-grey background
    d     = _disp
    phase = d["phase"]

    if phase == "starting":
        _txt(img, "Starting...", 400, 360, 2.0, (200, 200, 200), 3)

    elif phase == "waiting":
        _txt(img, "Waiting for both hands...", 160, 360, 1.8, (200, 200, 0), 3)

    elif phase == "calibrating":
        _txt(img, "HOLD HOME POSITION",         160, 300, 2.2, (0, 220, 220), 4)
        _txt(img, f"Snapping home in  {d['countdown']} s",
             300, 420, 1.6, (255, 255, 255), 2)

    else:  # streaming
        mode_col = (0, 180, 0) if not d["sim"] else (180, 140, 0)
        mode_txt = "ROBOT" if not d["sim"] else "SIMULATION"
        _txt(img,
             f"PIPER TELEOP  [{mode_txt}]  frame {d['frame']}  scale {d['scale']:.2f}",
             20, 42, 0.9, mode_col, 2)
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
            _txt(img, f"gripper {s['gripper']:5.1f}%", bx, by, 0.85, (200, 200, 200), 2)
            _bar(img, bx + 230, by - 22, 360, 26, s["gripper"], ik_col)

            if i == 0:
                cv2.line(img, (638, 60), (638, OH - 20), (60, 60, 60), 1)

        footer = "Ctrl+C to stop  |  Restart to re-calibrate home"
        if d["sim"]:
            footer += "   [SIMULATION - no robot motion]"
        _txt(img, footer, 20, OH - 20, 0.6, (80, 80, 80), 1)

    return img


# ── Terminal display helpers ───────────────────────────────────────────────────

def print_arm(label, s):
    color     = GREEN if s["ok"] else RED
    tick      = "OK" if s["ok"] else "!!"
    track_dot = f"{GREEN}[on]{RESET}"   if s["tracking"] else f"{RED}[lost]{RESET}"
    print(f"  {label}  {track_dot}   IK: {color}{tick} {s['msg']}{RESET}")
    print(f"    dx {s['dx']:+7.2f} cm    dy {s['dy']:+7.2f} cm    dz {s['dz']:+7.2f} cm")
    print(f"    dr {s['dr']:+7.2f} deg   dp {s['dp']:+7.2f} deg   yaw {s['dyaw']:+7.2f} deg")
    print(f"    gripper  {s['gripper']:5.1f} %")


def countdown(seconds, label, streamer):
    """Display a live countdown in both terminal and overlay."""
    for i in range(seconds, 0, -1):
        _disp["countdown"] = i
        t_end = time.time() + 1.0
        while time.time() < t_end:
            if streamer is not None and _cv2_ok:
                streamer.update_frame(render_frame())
            print(f"\r  {label}  {i:2d} s ...  ", end="", flush=True)
            time.sleep(1 / 30.0)
    print()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args       = parse_args()
    scale      = max(0.1, min(2.0, args.scale))
    rot_scale  = max(0.0, min(1.0, args.rot_scale))
    with_robot = args.with_robot

    _disp["scale"] = scale
    _disp["sim"]   = not with_robot

    print()
    print("=" * 60)
    print("  VISION PRO TELEOP - DUAL PIPER HANGING ARMS")
    print("=" * 60)
    print(f"  Vision Pro : {args.ip}")
    print(f"  Mode       : {'ROBOT (CAN active)' if with_robot else 'SIMULATION  (no robot motion)'}")
    print(f"  Scale      : pos={scale}  rot={rot_scale}{'  (position only)' if rot_scale == 0 else ''}")
    print(f"  Deadband   : {DEADBAND_CM} cm / {DEADBAND_DEG} deg")
    if with_robot:
        print(f"  Speed      : {args.speed} %")
    print(f"  Overlay    : {'cv2 ready' if _cv2_ok else 'terminal only  (pip install opencv-python)'}")
    print()

    # ── CAN port resolution ──────────────────────────────────────────────────
    left_can = right_can = None
    if with_robot:
        left_can  = args.left
        right_can = args.right
        # Only auto-detect if NEITHER was specified on CLI
        if left_can is None and right_can is None:
            auto_l, auto_r = get_can_ports()
            left_can  = auto_l
            right_can = auto_r
        if not left_can and not right_can:
            print("  ERROR: Cannot determine any CAN ports.")
            print("  Run:  sudo bash start_teleop.sh")
            print("  Or:   --with_robot --left can0 --right can1")
            sys.exit(1)
        parts = []
        if left_can:  parts.append(f"LEFT={left_can}")
        if right_can: parts.append(f"RIGHT={right_can}")
        print(f"  CAN  : {'   '.join(parts)}")
        if not left_can or not right_can:
            missing = "RIGHT" if not right_can else "LEFT"
            print(f"  NOTE : {missing} arm not connected — tracking only")
        print()

    # ── Precompute robot home reference ────────────────────────────────────────
    # Both arms are mechanically identical (same robot model, same HOME_POSITION).
    # Each arm is controlled in its own base frame — the absolute world position
    # of each arm base doesn't matter; we only work with EE deltas from home.
    T_home, _      = forward_kinematics(HOME_POSITION)
    home_robot_pos = list(T_home[:3, 3])
    home_robot_rpy = list(rotation_to_euler(T_home[:3, :3]))

    # ── Connect Vision Pro ────────────────────────────────────────────────────
    _disp["phase"] = "starting"
    print("  Connecting to Vision Pro ...", end=" ", flush=True)
    try:
        streamer = VisionProStreamer(ip=args.ip)
    except Exception as e:
        print(f"FAILED\n  {e}")
        sys.exit(1)
    print("OK")
    print()

    # ── Start video overlay stream ────────────────────────────────────────────
    if _cv2_ok:
        streamer.configure_video(device=None, size=f"{OW}x{OH}", fps=DISPLAY_HZ)
        streamer.start_webrtc()
        streamer.update_frame(render_frame())
        print("  Video overlay started (visible in Tracking Streamer app)")
        print()

    # ── Wait for both hands to be visible ────────────────────────────────────
    _disp["phase"] = "waiting"
    print("  Waiting for both hands ...", end=" ", flush=True)
    while True:
        data = streamer.get_latest()
        if data is not None and data.left is not None and data.right is not None:
            break
        if _cv2_ok:
            streamer.update_frame(render_frame())
        time.sleep(1 / 30.0)
    print("OK")
    print()

    # ── Calibration countdown: capture hand home pose ─────────────────────────
    _disp["phase"] = "calibrating"
    print(f"  {YELLOW}Hold BOTH hands at your robot home position.{RESET}")
    print(f"  {YELLOW}Keep them perfectly still -- capturing hand home pose ...{RESET}")
    countdown(10, "Snapping home in", streamer if _cv2_ok else None)

    data   = streamer.get_latest()
    home_L = data.left.wrist.copy()    # 4x4 SE(3) — left  wrist at home
    home_R = data.right.wrist.copy()   # 4x4 SE(3) — right wrist at home

    _disp["phase"] = "streaming"
    print(f"  {GREEN}Hand home pose captured - teleoperation active.{RESET}")
    print()

    # ── Connect robot arms (robot mode only) ──────────────────────────────────
    left_ctrl = right_ctrl = None
    left_cmd  = right_cmd  = None

    if with_robot:
        from piper_core import PiperHangingController

        print("  Connecting arms ...")
        threads = []

        if left_can:
            left_ctrl = PiperHangingController(can_port=left_can)
            def _cl(): left_ctrl._conn_ok = left_ctrl.connect()
            t = threading.Thread(target=_cl); t.start(); threads.append(t)

        if right_can:
            right_ctrl = PiperHangingController(can_port=right_can)
            def _cr(): right_ctrl._conn_ok = right_ctrl.connect()
            t = threading.Thread(target=_cr); t.start(); threads.append(t)

        for t in threads: t.join()

        if left_ctrl and not getattr(left_ctrl, '_conn_ok', False):
            print("  LEFT arm failed to connect - aborting.")
            sys.exit(1)
        if right_ctrl and not getattr(right_ctrl, '_conn_ok', False):
            print("  RIGHT arm failed to connect - aborting.")
            if left_ctrl: left_ctrl.shutdown()
            sys.exit(1)

        if left_ctrl:  left_ctrl.speed  = args.speed
        if right_ctrl: right_ctrl.speed = args.speed

        print("  Homing arms ...")
        threads = []
        if left_ctrl:
            t = threading.Thread(target=left_ctrl.go_home); t.start(); threads.append(t)
        if right_ctrl:
            t = threading.Thread(target=right_ctrl.go_home); t.start(); threads.append(t)
        for t in threads: t.join()
        print("  Arms ready.")
        print()

        # Non-blocking commanders — main loop never stalls on CAN
        if left_ctrl:  left_cmd  = ArmCommander(left_ctrl,  "left")
        if right_ctrl: right_cmd = ArmCommander(right_ctrl, "right")

    # ── IK warm-start joint tracker ───────────────────────────────────────────
    left_joints  = list(HOME_POSITION)
    right_joints = list(HOME_POSITION)

    last_grip_L = 100.0
    last_grip_R = 100.0

    frame        = 0
    last_display = 0.0

    sim_note = f"  {YELLOW}[SIMULATION - no robot motion]{RESET}" if not with_robot else ""
    print(f"  Streaming -- Ctrl+C to stop  |  Restart to re-calibrate{sim_note}")
    print()

    # ── Main control loop ─────────────────────────────────────────────────────
    try:
        while True:
            t_start = time.time()

            data     = streamer.get_latest()
            left_ok  = data is not None and data.left  is not None
            right_ok = data is not None and data.right is not None

            # ── LEFT: 6DOF delta from hand home ───────────────────────────────
            if left_ok:
                cur_L    = data.left.wrist
                # Position delta (metres) remapped to robot frame
                raw_dp_L = remap((cur_L[:3, 3] - home_L[:3, 3]) * scale, AXIS_MAP_L)
                dx_L = deadband(raw_dp_L[0] * 100, DEADBAND_CM)   # store in cm
                dy_L = deadband(raw_dp_L[1] * 100, DEADBAND_CM)
                dz_L = deadband(raw_dp_L[2] * 100, DEADBAND_CM)

                # Rotation delta: R_delta = R_cur @ R_home.T
                Rd_L       = rotation_delta(cur_L[:3, :3], home_L[:3, :3])
                rL, pL, yL = rotation_to_euler(Rd_L)
                dr_L   = deadband(math.degrees(rL) * rot_scale, DEADBAND_DEG)
                dp_L   = deadband(math.degrees(pL) * rot_scale, DEADBAND_DEG)
                dyaw_L = deadband(math.degrees(yL) * rot_scale, DEADBAND_DEG)

                pinch_L   = data.left.pinch_distance
                gripper_L = min(100.0, pinch_L / PINCH_MAX_M * 100)
            else:
                dx_L = dy_L = dz_L = dr_L = dp_L = dyaw_L = 0.0
                gripper_L = last_grip_L   # hold last known value when hand lost

            # ── RIGHT: 6DOF delta from hand home ──────────────────────────────
            if right_ok:
                cur_R    = data.right.wrist
                raw_dp_R = remap((cur_R[:3, 3] - home_R[:3, 3]) * scale, AXIS_MAP_R)
                dx_R = deadband(raw_dp_R[0] * 100, DEADBAND_CM)
                dy_R = deadband(raw_dp_R[1] * 100, DEADBAND_CM)
                dz_R = deadband(raw_dp_R[2] * 100, DEADBAND_CM)

                Rd_R       = rotation_delta(cur_R[:3, :3], home_R[:3, :3])
                rR, pR, yR = rotation_to_euler(Rd_R)
                dr_R   = deadband(math.degrees(rR) * rot_scale, DEADBAND_DEG)
                dp_R   = deadband(math.degrees(pR) * rot_scale, DEADBAND_DEG)
                dyaw_R = deadband(math.degrees(yR) * rot_scale, DEADBAND_DEG)

                pinch_R   = data.right.pinch_distance
                gripper_R = min(100.0, pinch_R / PINCH_MAX_M * 100)
            else:
                dx_R = dy_R = dz_R = dr_R = dp_R = dyaw_R = 0.0
                gripper_R = last_grip_R

            # ── IK validation — both arms, every frame ────────────────────────
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

            # Advance warm-start only when IK succeeds
            if lok and lq is not None:
                left_joints = lq
            if rok and rq is not None:
                right_joints = rq

            # ── Robot commands — non-blocking, truly parallel ─────────────────
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

                # Gripper — deadband-gated to avoid CAN spam
                if left_cmd is not None and abs(gripper_L - last_grip_L) >= GRIPPER_DEADBAND:
                    left_cmd.gripper(gripper_L / 100.0 * GRIPPER_MAX_MM)
                    last_grip_L = gripper_L

                if right_cmd is not None and abs(gripper_R - last_grip_R) >= GRIPPER_DEADBAND:
                    right_cmd.gripper(gripper_R / 100.0 * GRIPPER_MAX_MM)
                    last_grip_R = gripper_R

            frame += 1

            # ── Display update — throttled to DISPLAY_HZ ──────────────────────
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
                    streamer.update_frame(render_frame())

                sim_label = f"   {YELLOW}[SIM]{RESET}" if not with_robot else ""
                print(CLEAR, end="")
                print(f"  VISION PRO TELEOP   frame={frame}   "
                      f"scale={scale:.2f}   deadband={DEADBAND_CM}cm{sim_label}")
                print()
                print_arm("LEFT ", _disp["left"])
                print()
                print_arm("RIGHT", _disp["right"])
                print()
                print("  Ctrl+C to stop  |  Restart to re-calibrate home")

            # ── Rate limit ────────────────────────────────────────────────────
            elapsed = time.time() - t_start
            sleep   = 1.0 / POLL_HZ - elapsed
            if sleep > 0:
                time.sleep(sleep)

    except KeyboardInterrupt:
        print("\n\n  Stopped.")

    finally:
        # Stop ArmCommander background threads
        if left_cmd  is not None: left_cmd.stop()
        if right_cmd is not None: right_cmd.stop()

        # Relax and disconnect arms
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


if __name__ == "__main__":
    main()
