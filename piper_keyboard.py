#!/usr/bin/env python3
"""
Keyboard Controller for Piper Arm
===================================
Move the robot 1cm per keypress using keyboard keys.
Imports PiperDirectController from piper_direct.py.

Usage:
    python3 piper_keyboard.py
    python3 piper_keyboard.py --can can1
"""

import sys
import tty
import termios
from piper_direct import PiperDirectController, HOME_POSITION

STEP = 0.01  # 1cm in meters

# key -> (dx, dy, dz) direction
KEY_MAP = {
    'w': ( 1,  0,  0),   # +X forward
    's': (-1,  0,  0),   # -X backward
    'a': ( 0,  1,  0),   # +Y left
    'd': ( 0, -1,  0),   # -Y right
    'q': ( 0,  0,  1),   # +Z up
    'e': ( 0,  0, -1),   # -Z down
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
    pos, _ = ctrl.read_pose()
    print(
        f"  Pose: x={pos[0]*100:6.1f}  y={pos[1]*100:6.1f}  z={pos[2]*100:6.1f} cm"
        f"  |  speed={ctrl.speed}%"
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

    ctrl = PiperDirectController(can_port=can_port)
    if not ctrl.connect():
        print("Failed to connect.")
        sys.exit(1)

    ctrl.go_home()

    print()
    print("=" * 50)
    print("  KEYBOARD CONTROL  (1cm per press)")
    print("=" * 50)
    print("  w/s    +X / -X")
    print("  a/d    +Y / -Y")
    print("  q/e    +Z / -Z")
    print("  r      home")
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

            elif key == 'r':
                ctrl.go_home()
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
        print("  Going home ...", end=" ", flush=True)
        try:
            ctrl.send_joints(HOME_POSITION, timeout=8.0)
            print("Done")
        except Exception:
            print("Skipped")
        ctrl.shutdown()


if __name__ == "__main__":
    main()
