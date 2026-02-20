#!/usr/bin/env python3
"""
Piper Dual Arm Hanging Controller
===================================
Controls two hanging Piper arms (LEFT and RIGHT) simultaneously.
Sends the same commands to both arms in parallel.

Requires: sudo bash piper_setup.sh  (run once to identify arms)

Usage:
    python3 piper_hanging_dual.py
    python3 piper_hanging_dual.py --left can0 --right can1   # Override
"""

import sys
import os
import time
import math
import threading
from piper_hanging import (
    PiperHangingController,
    HOME_POSITION, RELAX_POSITION, SAFE_POSITION,
    JOINT_LOWER, JOINT_UPPER,
    verify_fk,
)


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


def run_parallel(left_ctrl, right_ctrl, func_name, *args, **kwargs):
    """Call a method on both controllers in parallel."""
    results = [None, None]

    def run_left():
        results[0] = getattr(left_ctrl, func_name)(*args, **kwargs)

    def run_right():
        results[1] = getattr(right_ctrl, func_name)(*args, **kwargs)

    t1 = threading.Thread(target=run_left)
    t2 = threading.Thread(target=run_right)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    return results


def print_both_poses(left, right):
    lpos, lrpy = left.read_pose()
    rpos, rrpy = right.read_pose()
    print(f"  LEFT:  x={lpos[0]*100:6.2f}  y={lpos[1]*100:6.2f}  z={lpos[2]*100:6.2f} cm")
    print(f"  RIGHT: x={rpos[0]*100:6.2f}  y={rpos[1]*100:6.2f}  z={rpos[2]*100:6.2f} cm")


def print_both_joints(left, right):
    lq = left.read_joints()
    rq = right.read_joints()
    ld = [math.degrees(a) for a in lq]
    rd = [math.degrees(a) for a in rq]
    print(f"  LEFT:  {' '.join(f'{d:7.1f}°' for d in ld)}")
    print(f"  RIGHT: {' '.join(f'{d:7.1f}°' for d in rd)}")


def interactive_mode(left, right):
    print()
    print("=" * 55)
    print("  PIPER DUAL ARM HANGING CONTROLLER")
    print("=" * 55)
    print()
    print(f"  Home (deg): {' '.join(f'{math.degrees(h):6.1f}' for h in HOME_POSITION)}")
    print()
    print("Commands (sent to BOTH arms):")
    print("  dx dy dz        Move by delta (cm)")
    print("  goto x y z      Move to position (cm)")
    print("  home            Go to mid-range home")
    print("  relax           Hang arms down")
    print("  pose            Print EE poses")
    print("  joints          Print joint angles")
    print("  speed 1-100     Set speed")
    print("  gripper open    Open grippers")
    print("  gripper close   Close grippers")
    print("  gripper <mm>    Set gripper opening")
    print("  quit            Exit")
    print()
    print("=" * 55)

    while True:
        try:
            user_input = input("\n[dual]> ").strip()
            if not user_input:
                continue

            parts = user_input.split()
            cmd = parts[0].lower()

            if cmd in ("quit", "q", "exit"):
                break

            elif cmd == "home":
                t1 = threading.Thread(target=left.go_home)
                t2 = threading.Thread(target=right.go_home)
                t1.start()
                t2.start()
                t1.join()
                t2.join()

            elif cmd == "relax":
                t1 = threading.Thread(target=left.go_relax)
                t2 = threading.Thread(target=right.go_relax)
                t1.start()
                t2.start()
                t1.join()
                t2.join()

            elif cmd == "pose":
                print_both_poses(left, right)

            elif cmd == "joints":
                print_both_joints(left, right)

            elif cmd == "speed":
                if len(parts) != 2:
                    print("Usage: speed 1-100")
                    continue
                s = max(1, min(100, int(float(parts[1]))))
                left.speed = s
                right.speed = s
                print(f"  Speed set to {s}% (both arms)")

            elif cmd == "gripper":
                if len(parts) != 2:
                    print("Usage: gripper open|close|<mm>")
                    continue
                arg = parts[1].lower()
                if arg == "open":
                    mm = 70
                elif arg == "close":
                    mm = 0
                else:
                    mm = float(arg)
                t1 = threading.Thread(target=left.gripper_ctrl, args=(mm,))
                t2 = threading.Thread(target=right.gripper_ctrl, args=(mm,))
                t1.start()
                t2.start()
                t1.join()
                t2.join()
                print(f"  Gripper: {mm:.0f}mm (both)")

            elif cmd == "goto":
                if len(parts) == 4:
                    x = float(parts[1]) / 100.0
                    y = float(parts[2]) / 100.0
                    z = float(parts[3]) / 100.0
                    t1 = threading.Thread(target=left.move_to, args=(x, y, z))
                    t2 = threading.Thread(target=right.move_to, args=(x, y, z))
                    t1.start()
                    t2.start()
                    t1.join()
                    t2.join()
                else:
                    print("Usage: goto x y z")

            else:
                # Delta move
                if len(parts) == 3:
                    dx = float(parts[0]) / 100.0
                    dy = float(parts[1]) / 100.0
                    dz = float(parts[2]) / 100.0
                    t1 = threading.Thread(target=left.move_delta, args=(dx, dy, dz))
                    t2 = threading.Thread(target=right.move_delta, args=(dx, dy, dz))
                    t1.start()
                    t2.start()
                    t1.join()
                    t2.join()
                elif len(parts) == 1:
                    dx = float(parts[0]) / 100.0
                    t1 = threading.Thread(target=left.move_delta, args=(dx, 0, 0))
                    t2 = threading.Thread(target=right.move_delta, args=(dx, 0, 0))
                    t1.start()
                    t2.start()
                    t1.join()
                    t2.join()
                else:
                    print("Enter: dx dy dz  (e.g., 5 0 3)")

        except ValueError:
            print("Invalid input. Enter numbers like: 5 0 3")
        except KeyboardInterrupt:
            print()
            break
        except Exception as e:
            print(f"Error: {e}")


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
        print("  Or:   python3 piper_hanging_dual.py --left can0 --right can1")
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
        print(f"  LEFT arm failed to connect!")
        sys.exit(1)
    if not results[1]:
        print(f"  RIGHT arm failed to connect!")
        left.shutdown()
        sys.exit(1)

    # Home both arms in parallel
    print()
    t1 = threading.Thread(target=left.go_home)
    t2 = threading.Thread(target=right.go_home)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    try:
        interactive_mode(left, right)
    finally:
        # Relax both arms before exit
        print("  Relaxing both arms ...")
        try:
            t1 = threading.Thread(target=left.go_relax)
            t2 = threading.Thread(target=right.go_relax)
            t1.start()
            t2.start()
            t1.join()
            t2.join()
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
