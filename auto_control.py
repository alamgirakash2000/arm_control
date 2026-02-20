#!/usr/bin/env python3
"""
Auto Control — send automated moves to the Piper arm.

Usage:
    python3 auto_control.py              # run the default sequence
    python3 auto_control.py --random     # random moves in a loop
"""

import sys
import time
import random
import rclpy
from piper_control import PiperDeltaController, HOME_POSITION


def random_moves(controller, interval=2.0, count=10):
    """Send random delta moves with a pause between each."""
    for i in range(count):
        dx = random.uniform(-0.05, 0.05)
        dy = random.uniform(-0.05, 0.05)
        dz = random.uniform(-0.03, 0.03)
        print(f"\n[{i+1}/{count}]")
        controller.move_delta(dx, dy, dz)
        time.sleep(interval)


def sequence_moves(controller):
    """Run a fixed sequence of goto positions."""
    targets = [
        (0.20, 0.00, 0.25),
        (0.25, 0.05, 0.20),
        (0.15, -0.05, 0.30),
    ]

    for i, (x, y, z) in enumerate(targets):
        print(f"\n[{i+1}/{len(targets)}]")
        controller.move_to(x, y, z)
        time.sleep(1.0)


def main():
    rclpy.init()
    controller = PiperDeltaController()
    controller.go_home()

    try:
        if "--random" in sys.argv:
            random_moves(controller, interval=2.0, count=10)
        else:
            sequence_moves(controller)
    finally:
        print("\n  Going home before exit ...", end=" ", flush=True)
        try:
            controller.moveit2.move_to_configuration(HOME_POSITION)
            controller.moveit2.wait_until_executed()
            print("Done")
        except Exception:
            print("Skipped")
        controller.shutdown()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
