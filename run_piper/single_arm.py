#!/usr/bin/env python3
"""
Piper Single Arm — Interactive controller.

Usage:
    python run_piper/single_arm.py --can can0
    python run_piper/single_arm.py --can can0 5 0 3   # one-shot delta (cm)
"""

import sys
import math
from piper_core import (
    PiperHangingController,
    HOME_POSITION, RELAX_POSITION, SAFE_POSITION, JOINT_LOWER,
    verify_fk,
)


def print_pose(ctrl):
    pos, rpy, _ = ctrl.get_pose()
    print(f"  Position:    x={pos[0]*100:.2f}  y={pos[1]*100:.2f}  z={pos[2]*100:.2f}  (cm)")
    print(f"  Orientation: r={math.degrees(rpy[0]):.1f}  p={math.degrees(rpy[1]):.1f}  y={math.degrees(rpy[2]):.1f}  (deg)")


def print_joints(ctrl):
    q = ctrl.read_joints()
    print(f"  Joints (deg): {' '.join(f'{math.degrees(a):8.2f}' for a in q)}")
    print(f"  Joints (rad): {' '.join(f'{a:8.4f}' for a in q)}")


def interactive_mode(ctrl):
    print()
    print("=" * 55)
    print("  PIPER ARM HANGING CONTROLLER")
    print("=" * 55)
    print()
    print(f"  Home joints (deg): {' '.join(f'{math.degrees(h):6.1f}' for h in HOME_POSITION)}")
    print()
    print("Commands:")
    print("  dx dy dz              Move by delta (cm)")
    print("  goto x y z            Move to absolute position (cm)")
    print("  goto x y z r p yaw    With orientation (degrees)")
    print("  home                  Go to mid-range home")
    print("  relax                 Hang arm down")
    print("  pose                  Print EE pose")
    print("  joints                Print joint angles")
    print("  speed 1-100           Set speed")
    print("  gripper open|close|<mm>")
    print("  quit")
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
            elif cmd == "relax":
                ctrl.go_relax()
            elif cmd == "pose":
                print_pose(ctrl)
            elif cmd == "joints":
                print_joints(ctrl)
            elif cmd == "speed":
                if len(parts) != 2:
                    print("Usage: speed 1-100")
                    continue
                ctrl.speed = max(1, min(100, int(float(parts[1]))))
                print(f"  Speed set to {ctrl.speed}%")
            elif cmd == "gripper":
                if len(parts) != 2:
                    print("Usage: gripper open|close|<mm>")
                    continue
                arg = parts[1].lower()
                mm = 70 if arg == "open" else 0 if arg == "close" else float(arg)
                ctrl.gripper_ctrl(mm)
                print(f"  Gripper: {mm:.0f} mm")
            elif cmd == "goto":
                if len(parts) == 4:
                    ctrl.move_to(*(float(p) / 100.0 for p in parts[1:4]))
                elif len(parts) == 7:
                    x, y, z = (float(p) / 100.0 for p in parts[1:4])
                    r, p, yw = (math.radians(float(p)) for p in parts[4:7])
                    ctrl.move_to(x, y, z, r, p, yw)
                else:
                    print("Usage: goto x y z [roll pitch yaw]")
            else:
                if len(parts) == 3:
                    ctrl.move_delta(*(float(p) / 100.0 for p in parts))
                elif len(parts) == 1:
                    ctrl.move_delta(float(parts[0]) / 100.0, 0.0, 0.0)
                else:
                    print("Enter: dx dy dz  (e.g., 5 0 3)")

        except ValueError:
            print("Invalid input.")
        except KeyboardInterrupt:
            print()
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    can_port = "can0"
    positional = []
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--can" and i + 1 < len(sys.argv):
            can_port = sys.argv[i + 1]
            i += 2
        else:
            if not sys.argv[i].startswith("--"):
                positional.append(sys.argv[i])
            i += 1

    print(f"  Home position (deg): {' '.join(f'{math.degrees(h):.1f}' for h in HOME_POSITION)}")
    print("  Verifying FK ...", end=" ", flush=True)
    print("OK" if verify_fk() else "MISMATCH (proceeding)")

    ctrl = PiperHangingController(can_port=can_port)
    if not ctrl.connect():
        print("  Failed to connect.")
        sys.exit(1)

    ctrl.go_home()

    try:
        if len(positional) == 3:
            ctrl.move_delta(*(float(p) / 100.0 for p in positional))
        else:
            interactive_mode(ctrl)
    finally:
        ctrl.go_relax()
        print("  Shutting down ...", end=" ", flush=True)
        ctrl.shutdown()
        print("Done")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
