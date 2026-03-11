#!/usr/bin/env python3
"""
Dual Keyboard Controller for Hanging Piper Arms
==================================================
Move two Piper arms simultaneously using keyboard keys.
Both arms receive the same commands by default.
Hold 'v' prefix for LEFT-only, 'b' prefix for RIGHT-only.

Requires: sudo bash piper_setup.sh  (run once to identify arms)

Usage:
    python3 dual_keyboard.py
    python3 dual_keyboard.py --left can0 --right can1
"""

import sys
import os
import math
import tty
import termios
import threading
from run_piper.piper_core import (
    PiperHangingController,
    HOME_POSITION, RELAX_POSITION, SAFE_POSITION,
    JOINT_LOWER, JOINT_UPPER,
    verify_fk,
    ik_solve, euler_to_rotation, forward_kinematics, joints_in_limits,
)

STEP = 0.01  # 1cm in meters
ROT_STEP = math.radians(5)  # 5° per keypress

# key -> (dx, dy, dz) direction
KEY_MAP = {
    'w': ( 1,  0,  0),   # +X forward
    's': (-1,  0,  0),   # -X backward
    'a': ( 0,  1,  0),   # +Y left
    'd': ( 0, -1,  0),   # -Y right
    'q': ( 0,  0,  1),   # +Z up
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


def get_can_ports():
    """Read LEFT/RIGHT CAN assignment from piper_setup.sh output."""
    env_file = "/tmp/piper_arms.env"
    if not os.path.exists(env_file):
        return None, None
    ports = {}
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                ports[k] = v
    return ports.get("LEFT"), ports.get("RIGHT")


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


def run_both(left, right, method, *args, **kwargs):
    """Call a method on both controllers in parallel."""
    t1 = threading.Thread(target=getattr(left, method), args=args, kwargs=kwargs)
    t2 = threading.Thread(target=getattr(right, method), args=args, kwargs=kwargs)
    t1.start()
    t2.start()
    t1.join()
    t2.join()


def show_poses(left, right):
    lpos, lrpy = left.read_pose()
    rpos, rrpy = right.read_pose()
    lr, lp, ly = [math.degrees(a) for a in lrpy]
    rr, rp, ry = [math.degrees(a) for a in rrpy]
    print(
        f"  L: x={lpos[0]*100:5.1f} y={lpos[1]*100:5.1f} z={lpos[2]*100:5.1f} cm"
        f"  RPY: {lr:5.1f} {lp:5.1f} {ly:5.1f}°"
        f"  | spd={left.speed}%"
    )
    print(
        f"  R: x={rpos[0]*100:5.1f} y={rpos[1]*100:5.1f} z={rpos[2]*100:5.1f} cm"
        f"  RPY: {rr:5.1f} {rp:5.1f} {ry:5.1f}°"
        f"  | spd={right.speed}%"
    )


def move_delta_pos(ctrl, dx, dy, dz):
    """Position-only delta move."""
    ctrl.move_delta(dx * STEP, dy * STEP, dz * STEP)


def move_delta_rot(ctrl, dr, dp, dy):
    """Rotation-only delta move."""
    pos, rpy = ctrl.read_pose()
    ctrl.move_to(
        pos[0], pos[1], pos[2],
        rpy[0] + dr * ROT_STEP,
        rpy[1] + dp * ROT_STEP,
        rpy[2] + dy * ROT_STEP,
    )


def apply_to(targets, func, *args):
    """Apply func to one or two controllers in parallel."""
    if len(targets) == 1:
        func(targets[0], *args)
    else:
        t1 = threading.Thread(target=func, args=(targets[0], *args))
        t2 = threading.Thread(target=func, args=(targets[1], *args))
        t1.start()
        t2.start()
        t1.join()
        t2.join()


def main():
    # Parse args
    left_can = None
    right_can = None
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--left" and i + 1 < len(sys.argv):
            left_can = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--right" and i + 1 < len(sys.argv):
            right_can = sys.argv[i + 1]
            i += 2
        else:
            i += 1

    # Auto-detect if not specified
    if left_can is None or right_can is None:
        auto_left, auto_right = get_can_ports()
        if left_can is None:
            left_can = auto_left
        if right_can is None:
            right_can = auto_right

    if not left_can or not right_can:
        print("  ERROR: Could not determine LEFT/RIGHT CAN ports.")
        print("  Run:  sudo bash piper_setup.sh")
        print("  Or:   python3 dual_keyboard.py --left can0 --right can1")
        sys.exit(1)

    print("=" * 55)
    print("  PIPER DUAL ARM CONTROLLER")
    print(f"  LEFT:  {left_can}")
    print(f"  RIGHT: {right_can}")
    print("=" * 55)
    print()

    # Verify FK
    print("  Verifying FK ...", end=" ", flush=True)
    if verify_fk():
        print("OK")
    else:
        print("MISMATCH (proceeding)")

    # Create controllers
    left = PiperHangingController(can_port=left_can)
    right = PiperHangingController(can_port=right_can)

    # Connect both in parallel
    print("  Connecting both arms ...")
    results = [False, False]

    def connect_left():
        print(f"    LEFT ({left_can}): ", end="", flush=True)
        results[0] = left.connect()

    def connect_right():
        print(f"    RIGHT ({right_can}): ", end="", flush=True)
        results[1] = right.connect()

    t1 = threading.Thread(target=connect_left)
    t2 = threading.Thread(target=connect_right)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    if not results[0]:
        print("  LEFT arm failed to connect!")
        sys.exit(1)
    if not results[1]:
        print("  RIGHT arm failed to connect!")
        left.shutdown()
        sys.exit(1)

    # Home both arms in parallel
    print()
    run_both(left, right, "go_home")

    print()
    print("=" * 55)
    print("  DUAL KEYBOARD CONTROL  (1cm / 5° per press)")
    print("=" * 55)
    print("  w/s    +X / -X  (forward / backward)")
    print("  a/d    +Y / -Y  (left / right)")
    print("  q/e    +Z / -Z  (up / down)")
    print("  i/k    +pitch / -pitch")
    print("  j/l    +yaw / -yaw")
    print("  n/m    +roll / -roll")
    print()
    print("  v+key  LEFT arm only   (press v, then the key)")
    print("  b+key  RIGHT arm only  (press b, then the key)")
    print()
    print("  r      home both arms")
    print("  t      relax both arms")
    print("  o/c    gripper open / close (both)")
    print("  [/]    speed down / up")
    print("  x      quit")
    print("=" * 55)
    print()
    show_poses(left, right)

    try:
        arm_filter = None  # None = both, 'left', 'right'

        while True:
            key = get_key()

            if key in ('\x1b', 'x', '\x03'):  # ESC, x, Ctrl+C
                break

            # Arm selection prefixes
            if key == 'v':
                arm_filter = 'left'
                continue
            if key == 'b':
                arm_filter = 'right'
                continue

            # Determine target controllers
            if arm_filter == 'left':
                targets = [left]
                tag = "[L]"
            elif arm_filter == 'right':
                targets = [right]
                tag = "[R]"
            else:
                targets = [left, right]
                tag = "[LR]"

            # Reset filter after use
            arm_filter = None

            if key in KEY_MAP:
                dx, dy, dz = KEY_MAP[key]
                apply_to(targets, move_delta_pos, dx, dy, dz)
                print(f"  {tag}", end=" ")
                show_poses(left, right)

            elif key in ROT_KEY_MAP:
                dr, dp, dy = ROT_KEY_MAP[key]
                apply_to(targets, move_delta_rot, dr, dp, dy)
                print(f"  {tag}", end=" ")
                show_poses(left, right)

            elif key == 'r':
                print(f"  {tag} Homing ...")
                if len(targets) == 2:
                    run_both(left, right, "go_home")
                else:
                    targets[0].go_home()
                show_poses(left, right)

            elif key == 't':
                print(f"  {tag} Relaxing ...")
                if len(targets) == 2:
                    run_both(left, right, "go_relax")
                else:
                    targets[0].go_relax()
                show_poses(left, right)

            elif key == 'o':
                if len(targets) == 2:
                    run_both(left, right, "gripper_ctrl", 70)
                else:
                    targets[0].gripper_ctrl(70)
                print(f"  {tag} Gripper opened")

            elif key == 'c':
                if len(targets) == 2:
                    run_both(left, right, "gripper_ctrl", 0)
                else:
                    targets[0].gripper_ctrl(0)
                print(f"  {tag} Gripper closed")

            elif key == ']':
                for t in targets:
                    t.speed = min(100, t.speed + 10)
                print(f"  {tag} Speed: L={left.speed}% R={right.speed}%")

            elif key == '[':
                for t in targets:
                    t.speed = max(1, t.speed - 10)
                print(f"  {tag} Speed: L={left.speed}% R={right.speed}%")

    finally:
        print()
        print("  Relaxing both arms ...")
        try:
            run_both(left, right, "go_relax")
        except Exception:
            print("  Skipped")

        print("  Shutting down ...", end=" ", flush=True)
        left.shutdown()
        right.shutdown()
        print("Done")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
