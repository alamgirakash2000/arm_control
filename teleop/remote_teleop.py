#!/usr/bin/env python3
"""
Remote Teleop Receiver — Dual Piper Hanging Arms
==================================================
Receives VR controller tracking data over HTTP from vr_sender.py
and drives two hanging Piper robot arms.

This script embeds its own HTTP relay server — no separate server process
needed.  Just run this on Ubuntu, then run vr_sender.py on your Windows
laptop (with SteamVR + Quest).

The architecture:
    Quest 3 → SteamVR (Windows) → vr_sender.py
                                      ↓  HTTP POST /pose @ 20 Hz
                                  remote_teleop.py (Ubuntu)
                                      ↓  shared memory (no HTTP overhead)
                                  absolute positions → CAN → Piper arms

Robot command flow:
    1. Robot homes → read actual physical pose → that's robot_home
    2. VR snaps → controller position at that moment → that's vr_home
    3. Every 1s:  target = robot_home + (vr_current - vr_home) * scale
    4. send_cartesian(target)  ← absolute position, robot goes there

Usage:
    python3 remote_teleop.py                                      # simulation
    python3 remote_teleop.py --with_robot                         # real robot
    python3 remote_teleop.py --with_robot --left can0 --right can1
"""

import argparse
import os
import sys
import time
import threading

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from piper_core import (
    rotation_to_euler,
    forward_kinematics,
    HOME_POSITION,
)

from relay_server import pose_store, start_server

try:
    import cv2
    _cv2_ok = True
except ImportError:
    _cv2_ok = False


# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_SCALE     = 1
DEFAULT_SPEED     = 70
DEFAULT_SMOOTHING = 0.9
GRIPPER_MAX_MM    = 70.0
GRIPPER_DEADBAND  = 2.0     # %
DEADBAND_CM       = 0.0     # cm — per-axis EMA deadband (0 = disabled, threshold handles jitter)
MOVE_THRESH_CM    = 2.0     # cm — per-axis must change by this much to accept new target
UPDATE_INTERVAL   = 0.1     # seconds between checking for new targets (10 Hz)
HOLD_HZ           = 30      # Hz — rate to resend current target (keeps robot locked)
DISPLAY_HZ        = 1       # terminal print rate
REF_SECS          = 5.0     # seconds to average for home pose capture
STALE_MS          = 500.0   # pose data older than this = tracking lost

# ── Axis remapping ────────────────────────────────────────────────────────────
# OpenVR frame: X=right, Y=up, -Z=forward
AXIS_MAP = [(1,  1), (0,  1), (2, -1)]

# ── ANSI helpers ──────────────────────────────────────────────────────────────
GREEN  = "\033[32m"
RED    = "\033[31m"
YELLOW = "\033[33m"
RESET  = "\033[0m"

# ── cv2 overlay ──────────────────────────────────────────────────────────────
OW, OH = 1280, 720
_disp = {
    "phase": "starting", "countdown": 0, "frame": 0,
    "scale": DEFAULT_SCALE, "sim": True, "server": "", "age_ms": -1,
    "left":  dict(tracking=False, dx=0, dy=0, dz=0, gripper=100),
    "right": dict(tracking=False, dx=0, dy=0, dz=0, gripper=100),
}


# ── Non-blocking robot commander ─────────────────────────────────────────────
# Each arm gets its own background thread that resends the current target
# at HOLD_HZ using raw CAN commands (MotionCtrl_2 + EndPoseCtrl).
# The main loop just updates the target; the thread handles continuous sending.

import math as _math

class ArmCommander:
    def __init__(self, ctrl, label="arm"):
        self._piper   = ctrl.piper   # direct SDK object
        self._speed   = ctrl.speed
        self._lock    = threading.Lock()
        self._target  = None         # (x, y, z, roll, pitch, yaw) or None
        self._grip_mm = None
        self._running = True
        self._label   = label
        threading.Thread(target=self._loop, daemon=True, name=f"cmd-{label}").start()

    def send(self, x, y, z, roll, pitch, yaw):
        """Update the target pose (called from main loop)."""
        with self._lock:
            self._target = (x, y, z, roll, pitch, yaw)

    def gripper(self, mm: float):
        """Update the gripper target."""
        with self._lock:
            self._grip_mm = float(mm)

    def stop(self):
        self._running = False

    def _loop(self):
        """Background loop: resend current target at HOLD_HZ."""
        interval = 1.0 / HOLD_HZ
        while self._running:
            with self._lock:
                tgt = self._target
                grip = self._grip_mm
            if tgt is not None:
                try:
                    x, y, z, roll, pitch, yaw = tgt
                    self._piper.MotionCtrl_2(0x01, 0x00, self._speed, 0x00)
                    self._piper.EndPoseCtrl(
                        round(x * 1e6), round(y * 1e6), round(z * 1e6),
                        round(_math.degrees(roll)  * 1000),
                        round(_math.degrees(pitch) * 1000),
                        round(_math.degrees(yaw)   * 1000),
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
        description="Remote teleop receiver for dual Piper hanging arms.",
    )
    p.add_argument("--with_robot",     action="store_true")
    p.add_argument("--left",           default=None, metavar="IFACE")
    p.add_argument("--right",          default=None, metavar="IFACE")
    p.add_argument("--scale",          default=DEFAULT_SCALE, type=float)
    p.add_argument("--speed",          default=DEFAULT_SPEED, type=int)
    p.add_argument("--smoothing",      default=DEFAULT_SMOOTHING, type=float)
    p.add_argument("--ref-secs",       default=REF_SECS, type=float)
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


# ── Math helpers ──────────────────────────────────────────────────────────────

def rotation_delta(R_cur, R_home):
    return R_cur @ R_home.T

def deadband(v, threshold):
    return 0.0 if abs(v) < threshold else v

def remap(pos_delta_m, axis_map):
    return np.array([pos_delta_m[ax] * sign for ax, sign in axis_map])


# ── Reference pose capture ────────────────────────────────────────────────────

def capture_reference(secs, poll_interval):
    pos_samples = {"left": [], "right": []}
    R_samples   = {"left": [], "right": []}
    active_t    = 0.0
    t_wall0     = time.time()

    while active_t < secs:
        if time.time() - t_wall0 > 120.0:
            raise RuntimeError("Controllers not tracked for 2 min during reference capture.")
        frame = read_latest_pose()
        ok = {}
        for hand in ("left", "right"):
            pos, R, _, status = extract_controller(frame, hand)
            ok[hand] = status == "OK"
            if ok[hand]:
                pos_samples[hand].append(pos)
                R_samples[hand].append(R)
        if ok["left"] and ok["right"]:
            active_t += poll_interval
        rem = max(0.0, secs - active_t)
        print(f"\r  ref remaining: {rem:4.1f}s  "
              f"[L:{'OK' if ok['left'] else 'wait'} "
              f"R:{'OK' if ok['right'] else 'wait'}]   ", end="", flush=True)
        if _cv2_ok:
            cv2.imshow("Remote Teleop", render_frame())
            cv2.waitKey(1)
        time.sleep(max(0.0, poll_interval))

    print()
    out = {}
    for hand in ("left", "right"):
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
        _txt(img,
             f"REMOTE TELEOP [{mode_txt}]  frame {d['frame']}  "
             f"scale {d['scale']:.2f}  age {age_ms:.0f}ms",
             20, 42, 0.75, mode_col, 2)
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
            _txt(img, f"dx {s['dx']:+6.2f}cm", bx,       by, 0.85, (200, 200, 200), 2)
            _txt(img, f"dy {s['dy']:+6.2f}cm", bx + 180, by, 0.85, (200, 200, 200), 2)
            _txt(img, f"dz {s['dz']:+6.2f}cm", bx + 360, by, 0.85, (200, 200, 200), 2)

            by += 44
            _txt(img, f"gripper {s['gripper']:5.1f}%  [trigger]",
                 bx, by, 0.85, (200, 200, 200), 2)
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
                cv2.imshow("Remote Teleop", render_frame())
                cv2.waitKey(1)
            print(f"\r  {label}  {i:2d} s ...  ", end="", flush=True)
            time.sleep(1 / 30.0)
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args        = parse_args()
    scale       = max(0.1, min(2.0, args.scale))
    with_robot  = args.with_robot
    alpha       = float(np.clip(args.smoothing, 0.0, 1.0))
    invert_grip = bool(args.invert_gripper)

    _disp["scale"]  = scale
    _disp["sim"]    = not with_robot
    _disp["server"] = f"{args.host}:{args.port}"

    print()
    print("=" * 62)
    print("  REMOTE TELEOP — DUAL PIPER HANGING ARMS")
    print("=" * 62)
    print(f"  Mode      : {'ROBOT' if with_robot else 'SIMULATION'}")
    print(f"  Scale     : {scale}    Hold: {HOLD_HZ} Hz   Update: every {UPDATE_INTERVAL}s   Thresh: {MOVE_THRESH_CM} cm")
    print(f"  Smoothing : {alpha}")
    if with_robot:
        print(f"  Speed     : {args.speed} %")
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
            sys.exit(1)
        print(f"  CAN  : LEFT={left_can}   RIGHT={right_can}")
        print()

    # ── Start embedded HTTP relay server ──────────────────────────────────────
    _disp["phase"] = "starting"
    if _cv2_ok:
        cv2.imshow("Remote Teleop", render_frame())
        cv2.waitKey(1)

    print(f"  Starting HTTP server on {args.host}:{args.port} ...")
    start_server(host=args.host, port=args.port, blocking=False)
    print(f"  Server ready. On Windows run:")
    print(f"    python vr_sender.py --server http://10.46.34.149:{args.port}")
    print()

    # ── Wait for VR sender data ───────────────────────────────────────────────
    _disp["phase"] = "waiting"
    print("  Waiting for tracking data ...", end=" ", flush=True)

    t_wait0 = time.time()
    while True:
        frame = read_latest_pose()
        if frame is not None:
            _, _, _, ls = extract_controller(frame, "left")
            _, _, _, rs = extract_controller(frame, "right")
            if ls == "OK" and rs == "OK":
                break
        if time.time() - t_wait0 > 300.0:
            print("\n  TIMEOUT.")
            sys.exit(1)
        if _cv2_ok:
            cv2.imshow("Remote Teleop", render_frame())
            cv2.waitKey(1)
        time.sleep(0.1)

    print("OK — both controllers tracked!")
    print()

    # ── Capture VR home reference pose ────────────────────────────────────────
    _disp["phase"] = "calibrating"
    print(f"  {YELLOW}Hold BOTH controllers at robot home position.{RESET}")
    countdown_display(int(args.ref_secs) + 3, "Snapping home in")

    refs = capture_reference(secs=float(args.ref_secs), poll_interval=1.0 / HOLD_HZ)
    vr_home_pos_L, vr_home_R_L = refs["left"]
    vr_home_pos_R, vr_home_R_R = refs["right"]

    _disp["phase"] = "streaming"
    print(f"  {GREEN}VR home captured.{RESET}")
    print()

    # ── Connect robot arms & read actual home position ────────────────────────
    left_ctrl = right_ctrl = None
    left_cmd  = right_cmd  = None

    # Robot home = actual physical pose after homing (read from robot)
    # In simulation, use FK theoretical home
    T_home, _ = forward_kinematics(HOME_POSITION)
    robot_home_L = {"pos": list(T_home[:3, 3]), "rpy": list(rotation_to_euler(T_home[:3, :3]))}
    robot_home_R = {"pos": list(T_home[:3, 3]), "rpy": list(rotation_to_euler(T_home[:3, :3]))}

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
            print("  LEFT arm failed."); sys.exit(1)
        if not results[1]:
            print("  RIGHT arm failed."); left_ctrl.shutdown(); sys.exit(1)

        left_ctrl.speed  = args.speed
        right_ctrl.speed = args.speed

        print("  Homing both arms ...")
        t1 = threading.Thread(target=left_ctrl.go_home)
        t2 = threading.Thread(target=right_ctrl.go_home)
        t1.start(); t2.start(); t1.join(); t2.join()

        # READ ACTUAL ROBOT POSE — this is the real home position
        lpos, lrpy = left_ctrl.read_pose()
        rpos, rrpy = right_ctrl.read_pose()
        robot_home_L = {"pos": lpos, "rpy": lrpy}
        robot_home_R = {"pos": rpos, "rpy": rrpy}

        print(f"  LEFT  home: x={lpos[0]*100:.2f}  y={lpos[1]*100:.2f}  z={lpos[2]*100:.2f} cm")
        print(f"  RIGHT home: x={rpos[0]*100:.2f}  y={rpos[1]*100:.2f}  z={rpos[2]*100:.2f} cm")
        print()

        left_cmd  = ArmCommander(left_ctrl,  "left")
        right_cmd = ArmCommander(right_ctrl, "right")

    # ── State ─────────────────────────────────────────────────────────────────
    # "committed" = what the robot is currently holding (in cm, delta from home)
    com_L = [0.0, 0.0, 0.0]   # committed left  [dx, dy, dz] in cm
    com_R = [0.0, 0.0, 0.0]   # committed right  [dx, dy, dz] in cm
    com_grip_L = 100.0
    com_grip_R = 100.0
    # "candidate" = latest smoothed VR reading (may or may not be accepted)
    sdx_L = sdy_L = sdz_L = 0.0
    sdx_R = sdy_R = sdz_R = 0.0
    gripper_L = gripper_R = 100.0
    frame_num   = 0
    last_update = 0.0   # last time we checked whether to accept new target
    last_print  = 0.0

    sim_note = f"  {YELLOW}[SIMULATION]{RESET}" if not with_robot else ""
    print(f"  Streaming — Ctrl+C to stop{sim_note}")
    print()

    # Print header
    print(f"  {'time':>8s}  {'L_x':>7s} {'L_y':>7s} {'L_z':>7s}  "
          f"{'R_x':>7s} {'R_y':>7s} {'R_z':>7s}  "
          f"{'L_gr':>5s} {'R_gr':>5s}  {'age':>5s}  {'L_status':<14s} {'R_status':<14s}")
    print(f"  {'':->8s}  {'':->7s} {'':->7s} {'':->7s}  "
          f"{'':->7s} {'':->7s} {'':->7s}  "
          f"{'':->5s} {'':->5s}  {'':->5s}  {'':->14s} {'':->14s}")

    # ── Main control loop ─────────────────────────────────────────────────────
    try:
        while True:
            t_start = time.time()

            frame = read_latest_pose()
            age_ms = frame.get("_age_ms", -1) if frame else -1
            _disp["age_ms"] = age_ms

            # ── Read VR controllers → update candidate (smoothed) ────────────
            pos_L, R_L, trig_L, status_L = extract_controller(frame, "left")
            left_ok = status_L == "OK"
            if left_ok:
                raw_dp_L = remap((pos_L - vr_home_pos_L) * scale, AXIS_MAP)
                sdx_L = alpha * deadband(raw_dp_L[2] * 100, DEADBAND_CM) + (1 - alpha) * sdx_L
                sdy_L = alpha * deadband(raw_dp_L[1] * 100, DEADBAND_CM) + (1 - alpha) * sdy_L
                sdz_L = alpha * deadband(-raw_dp_L[0] * 100, DEADBAND_CM) + (1 - alpha) * sdz_L
                t_val_L   = 1.0 - trig_L if invert_grip else trig_L
                gripper_L = float(np.clip(t_val_L * 100.0, 0.0, 100.0))

            pos_R, R_R, trig_R, status_R = extract_controller(frame, "right")
            right_ok = status_R == "OK"
            if right_ok:
                raw_dp_R = remap((pos_R - vr_home_pos_R) * scale, AXIS_MAP)
                sdx_R = alpha * deadband(raw_dp_R[2] * 100, DEADBAND_CM) + (1 - alpha) * sdx_R
                sdy_R = alpha * deadband(raw_dp_R[1] * 100, DEADBAND_CM) + (1 - alpha) * sdy_R
                sdz_R = alpha * deadband(-raw_dp_R[0] * 100, DEADBAND_CM) + (1 - alpha) * sdz_R
                t_val_R   = 1.0 - trig_R if invert_grip else trig_R
                gripper_R = float(np.clip(t_val_R * 100.0, 0.0, 100.0))

            frame_num += 1

            # ── Every UPDATE_INTERVAL: accept new target only if moved enough ─
            now = time.time()
            if now - last_update >= UPDATE_INTERVAL:
                last_update = now
                cand_L = [sdx_L, sdy_L, sdz_L]
                cand_R = [sdx_R, sdy_R, sdz_R]
                # Per-axis: only update axes that changed >= threshold
                for i in range(3):
                    if abs(cand_L[i] - com_L[i]) >= MOVE_THRESH_CM:
                        com_L[i] = cand_L[i]
                for i in range(3):
                    if abs(cand_R[i] - com_R[i]) >= MOVE_THRESH_CM:
                        com_R[i] = cand_R[i]
                # Gripper always updates (no threshold beyond deadband)
                if abs(gripper_L - com_grip_L) >= GRIPPER_DEADBAND:
                    com_grip_L = gripper_L
                if abs(gripper_R - com_grip_R) >= GRIPPER_DEADBAND:
                    com_grip_R = gripper_R

            # ── Compute absolute target from committed deltas ────────────────
            target_L = [
                robot_home_L["pos"][0] + com_L[0] / 100,
                robot_home_L["pos"][1] + com_L[1] / 100,
                robot_home_L["pos"][2] + com_L[2] / 100,
            ]
            target_R = [
                robot_home_R["pos"][0] + com_R[0] / 100,
                robot_home_R["pos"][1] + com_R[1] / 100,
                robot_home_R["pos"][2] + com_R[2] / 100,
            ]

            # ── Update target for background threads (they resend at HOLD_HZ) ──
            if with_robot:
                if left_cmd is not None:
                    left_cmd.send(
                        target_L[0], target_L[1], target_L[2],
                        robot_home_L["rpy"][0], robot_home_L["rpy"][1], robot_home_L["rpy"][2],
                    )
                    left_cmd.gripper(com_grip_L / 100.0 * GRIPPER_MAX_MM)
                if right_cmd is not None:
                    right_cmd.send(
                        target_R[0], target_R[1], target_R[2],
                        robot_home_R["rpy"][0], robot_home_R["rpy"][1], robot_home_R["rpy"][2],
                    )
                    right_cmd.gripper(com_grip_R / 100.0 * GRIPPER_MAX_MM)

            # ── Print + cv2 update at DISPLAY_HZ (1 Hz) ─────────────────
            if now - last_print >= 1.0 / DISPLAY_HZ:
                last_print = now

                print(f"  {now - t_wait0:7.1f}s  "
                      f"{target_L[0]*100:+7.2f} {target_L[1]*100:+7.2f} {target_L[2]*100:+7.2f}  "
                      f"{target_R[0]*100:+7.2f} {target_R[1]*100:+7.2f} {target_R[2]*100:+7.2f}  "
                      f"{com_grip_L:5.1f} {com_grip_R:5.1f}  "
                      f"{age_ms:5.0f}ms  {status_L:<14s} {status_R:<14s}")

                _disp["frame"] = frame_num
                _disp["left"]  = dict(tracking=left_ok,
                                      dx=com_L[0], dy=com_L[1], dz=com_L[2],
                                      gripper=com_grip_L)
                _disp["right"] = dict(tracking=right_ok,
                                      dx=com_R[0], dy=com_R[1], dz=com_R[2],
                                      gripper=com_grip_R)

                if _cv2_ok:
                    cv2.imshow("Remote Teleop", render_frame())
                    if cv2.waitKey(1) == 27:
                        break

            # ── Rate limit main loop (VR reading + target update) ────────────
            elapsed = time.time() - t_start
            sleep = 0.05 - elapsed   # ~20 Hz for VR reading, plenty fast
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

            print("  Shutting down ...", end=" ", flush=True)
            try:
                if left_ctrl  is not None: left_ctrl.shutdown()
                if right_ctrl is not None: right_ctrl.shutdown()
                print("done")
            except Exception as e:
                print(f"skipped ({e})")

        if _cv2_ok:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
