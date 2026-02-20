#!/usr/bin/env python3
"""
Piper Arm Zero-Point Recalibration (All Joints)
=================================================
Disables all motors, gives you 10 seconds to straighten the arm
to the zero position, then automatically sets all zero points.

NO KEYBOARD INPUT NEEDED — just run and position the arm.

Usage:
    python3 piper_recal.py              # Uses can0
    python3 piper_recal.py --can can1   # Specify CAN port
"""

import sys
import time
import math
from piper_sdk import C_PiperInterface_V2

RAD_TO_MDEG = 1000.0 * 180.0 / math.pi


def read_joints(piper):
    msgs = piper.GetArmJointMsgs()
    js = msgs.joint_state
    mdeg = [js.joint_1, js.joint_2, js.joint_3,
            js.joint_4, js.joint_5, js.joint_6]
    return [math.degrees(v / RAD_TO_MDEG) for v in mdeg]


def main():
    can_port = "can0"
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--can" and i + 1 < len(sys.argv):
            can_port = sys.argv[i + 1]
            i += 2
        else:
            i += 1

    print("=" * 60)
    print("  PIPER ARM ZERO-POINT RECALIBRATION")
    print("=" * 60)
    print(f"  CAN port: {can_port}")
    print()

    piper = C_PiperInterface_V2(can_port)
    print("  Connecting ...", end=" ", flush=True)
    piper.ConnectPort()
    time.sleep(0.5)
    print("Done")
    print()

    # Show current readings
    print("  Current joint positions:")
    joints = read_joints(piper)
    for j in range(6):
        print(f"    Joint {j+1}: {joints[j]:8.2f}°")
    print()

    print("  WHAT WILL HAPPEN:")
    print("  1. All 6 motors will go COMPLETELY LIMP (no power)")
    print("  2. You get 10 seconds to straighten the arm to zero position")
    print("     - All segments in a straight line")
    print("     - Hanging: straight vertical line down from base")
    print("     - Align grooves/marks on each joint")
    print("  3. Zero points are set automatically — no input needed")
    print()
    print("  HOLD THE ARM — it will be heavy when motors go off!")
    print()
    print("  Starting in 15 seconds — go hold the arm!")
    for sec in range(15, 0, -1):
        print(f"  Motors off in {sec}s ...", end="\r", flush=True)
        time.sleep(1.0)
    print()

    # Disable ALL motors
    print()
    print("  >>> ALL MOTORS OFF — ARM IS LIMP <<<")
    for m in range(1, 7):
        piper.DisableArm(m)
        time.sleep(0.1)
    print()

    # 10 second countdown with live joint readings
    print("  Straighten the arm NOW! You have 10 seconds.")
    print()
    for sec in range(10, 0, -1):
        joints = read_joints(piper)
        vals = "  ".join(f"J{j+1}:{joints[j]:7.1f}°" for j in range(6))
        print(f"  [{sec:2d}s]  {vals}")
        time.sleep(1.0)

    print()
    print("  Time's up! Setting zero points ...")
    print()

    # Show final position
    joints = read_joints(piper)
    print("  Final position before setting zeros:")
    for j in range(6):
        print(f"    Joint {j+1}: {joints[j]:8.2f}°")
    print()

    # Set zero for all joints
    for m in range(1, 7):
        print(f"  Setting Joint {m} zero ...", end=" ", flush=True)
        piper.JointConfig(m, 0xAE)
        time.sleep(0.5)
        print("Done")

    time.sleep(1.0)

    # Re-enable all motors
    print()
    print("  Re-enabling motors ...", end=" ", flush=True)
    for m in range(1, 7):
        piper.EnableArm(m)
        time.sleep(0.1)
    time.sleep(0.5)
    print("Done")

    # Verify
    print()
    print("  Readings after recalibration (should all be near 0°):")
    joints = read_joints(piper)
    all_ok = True
    for j in range(6):
        ok = "OK" if abs(joints[j]) < 5.0 else "CHECK"
        if abs(joints[j]) >= 5.0:
            all_ok = False
        print(f"    Joint {j+1}: {joints[j]:8.2f}°  [{ok}]")
    print()

    if all_ok:
        print("  All joints calibrated!")
    else:
        print("  Some joints not near 0°. You may need to re-run.")
    print()

    # Try to enable
    print("  Attempting full arm enable ...", end=" ", flush=True)
    enabled = False
    timeout = time.time() + 10.0
    while time.time() < timeout:
        if piper.EnablePiper():
            enabled = True
            break
        time.sleep(0.01)

    if enabled:
        print("SUCCESS!")
        print()
        print("  Let the arm hang naturally and check:")
        print("  - J2 should read ~90°")
        print("  - J3 should read ~-170°")
        piper.DisablePiper()
    else:
        print("FAILED")
        print("  Run piper_diagnose.py for details.")

    print()
    print("=" * 60)
    print("  DONE — Zero points saved to motor flash memory")
    print("=" * 60)


if __name__ == "__main__":
    main()
