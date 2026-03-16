#!/usr/bin/env python3
"""
Keyboard Controller for Hanging Piper Arm
==========================================
Move the robot 1cm per keypress using keyboard keys.
Uses PiperHangingController with mid-range home and safe transitions.

Usage:
    python3 piper_keyboard_hanging.py
    python3 piper_keyboard_hanging.py --can can1
"""

import sys
import math
import tty
import termios
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "teleop"))
from piper_core import (
    PiperHangingController, HOME_POSITION, RELAX_POSITION, SAFE_POSITION,
    JOINT_LOWER,
)

STEP = 0.01  # 1cm in meters
ROT_STEP = math.radians(5)  # 5° per keypress

# key -> (dx, dy, dz) direction
KEY_MAP = {
    'w': ( 1,  0,  0),   # +X forward
    's': (-1,  0,  0),   # -X backward
    'a': ( 0,  1,  0),   # +Y left
    'd': ( 0, -1,  0),   # -Y right
    'q': ( 0,  0,  1),   # +Z up  (real world up, thanks to T_BASE flip)
    'e': ( 0,  0, -1),   # -Z down
}

# key -> (droll, dpitch, dyaw) direction
ROT_KEY_MAP = {
    'n': ( 1,  0,  0),   # +roll
    'm': (-1,  0,  0),   # -roll
    'i': ( 0,  1,  0),   # +pitch
    'k': ( 0, -1,  0),   # -pitch
    'j': ( 0,  0,  1),   # +yaw
    'l': ( 0,  0, -1),   # -yaw
}


def get_key():
    """Read a single keypress (no Enter needed)."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


def show_pose(ctrl):
    pos, rpy = ctrl.read_pose()
    r, p, y = [math.degrees(a) for a in rpy]
    print(
        f"  Pos: x={pos[0]*100:5.1f} y={pos[1]*100:5.1f} z={pos[2]*100:5.1f} cm"
        f"  RPY: {r:5.1f} {p:5.1f} {y:5.1f}°"
        f"  | spd={ctrl.speed}%"
    )


def main():
    can_port = "can0"
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--can" and i + 1 < len(sys.argv):
            can_port = sys.argv[i + 1]
            i += 2
            continue
        i += 1

    ctrl = PiperHangingController(can_port=can_port)
    if not ctrl.connect():
        print("Failed to connect.")
        sys.exit(1)

    ctrl.go_home()

    print()
    print("=" * 50)
    print("  KEYBOARD CONTROL — HANGING  (1cm per press)")
    print("=" * 50)
    print("  w/s    +X / -X  (forward / backward)")
    print("  a/d    +Y / -Y  (left / right)")
    print("  q/e    +Z / -Z  (up / down)")
    print("  i/k    +pitch / -pitch")
    print("  j/l    +yaw / -yaw")
    print("  n/m    +roll / -roll")
    print("  r      home     (safe -> mid-range)")
    print("  t      relax    (safe -> elbow -> hang)")
    print("  o/c    gripper open / close")
    print("  [/]    speed down / up")
    print("  x      quit")
    print("=" * 50)
    print()
    show_pose(ctrl)

    try:
        while True:
            key = get_key()

            if key in ('\x1b', 'x', '\x03'):  # ESC, x, Ctrl+C
                break

            elif key in KEY_MAP:
                dx, dy, dz = KEY_MAP[key]
                ctrl.move_delta(dx * STEP, dy * STEP, dz * STEP)
                show_pose(ctrl)

            elif key in ROT_KEY_MAP:
                dr, dp, dy = ROT_KEY_MAP[key]
                pos, rpy = ctrl.read_pose()
                ctrl.move_to(
                    pos[0], pos[1], pos[2],
                    rpy[0] + dr * ROT_STEP,
                    rpy[1] + dp * ROT_STEP,
                    rpy[2] + dy * ROT_STEP,
                )
                show_pose(ctrl)

            elif key == 'r':
                ctrl.go_home()
                show_pose(ctrl)

            elif key == 't':
                ctrl.go_relax()
                show_pose(ctrl)

            elif key == 'o':
                ctrl.gripper_ctrl(70)
                print("  Gripper opened")

            elif key == 'c':
                ctrl.gripper_ctrl(0)
                print("  Gripper closed")

            elif key == ']':
                ctrl.speed = min(100, ctrl.speed + 10)
                print(f"  Speed: {ctrl.speed}%")

            elif key == '[':
                ctrl.speed = max(1, ctrl.speed - 10)
                print(f"  Speed: {ctrl.speed}%")

    finally:
        print()
        print("  Relaxing before exit (safe -> elbow -> relax) ...", end=" ", flush=True)
        try:
            ctrl.send_joints(SAFE_POSITION, timeout=8.0)
            ctrl.send_joints([0.0, 0.0, JOINT_LOWER[2], 0.0, 0.0, 0.0], timeout=8.0)
            ctrl.send_joints(RELAX_POSITION, timeout=8.0)
            print("Done")
        except Exception:
            print("Skipped")
        ctrl.shutdown()


if __name__ == "__main__":
    main()
