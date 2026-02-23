#!/usr/bin/env python3
"""
Test a single joint (default: J3 elbow) by stepping through positions.
Reports commanded vs actual position at each step.

Usage:
    python3 test_elbow.py              # can0, joint 3
    python3 test_elbow.py --can can1   # can1, joint 3
    python3 test_elbow.py --can can1 --joint 2   # can1, joint 2
"""

import sys
import time
import math
from piper_sdk import C_PiperInterface_V2

RAD_TO_MDEG = 1000.0 * 180.0 / math.pi

# Joint limits (degrees) from URDF
LIMITS_DEG = [
    (-150.0, 150.0),   # J1
    (0.0, 180.0),      # J2
    (-170.0, 0.0),     # J3
    (-100.0, 100.0),   # J4
    (-70.0, 70.0),     # J5
    (-120.0, 120.0),   # J6
]


def read_joints_deg(piper):
    msgs = piper.GetArmJointMsgs()
    js = msgs.joint_state
    mdeg = [js.joint_1, js.joint_2, js.joint_3,
            js.joint_4, js.joint_5, js.joint_6]
    return [v / RAD_TO_MDEG * 180.0 / math.pi for v in mdeg]


def read_joint_deg(piper, joint_idx):
    """Read a single joint in degrees (0-indexed)."""
    all_j = read_joints_deg(piper)
    return all_j[joint_idx]


def send_joint_position(piper, joint_idx, target_deg, speed=30):
    """Send all joints to 0 except the target joint."""
    targets_deg = [0.0] * 6
    targets_deg[joint_idx] = target_deg
    cmds = [round(math.radians(d) * RAD_TO_MDEG) for d in targets_deg]

    piper.MotionCtrl_2(0x01, 0x01, speed, 0x00)
    piper.JointCtrl(*cmds)


def main():
    can_port = "can0"
    joint_num = 3  # 1-indexed (user-facing)
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--can" and i + 1 < len(sys.argv):
            can_port = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--joint" and i + 1 < len(sys.argv):
            joint_num = int(sys.argv[i + 1])
            i += 2
        else:
            i += 1

    joint_idx = joint_num - 1  # 0-indexed
    lo, hi = LIMITS_DEG[joint_idx]

    print("=" * 60)
    print(f"  JOINT {joint_num} MOTOR TEST  ({can_port})")
    print(f"  Range: {lo:.0f}° to {hi:.0f}°")
    print("=" * 60)
    print()

    piper = C_PiperInterface_V2(can_port)
    print("  Connecting ...", end=" ", flush=True)
    piper.ConnectPort()
    time.sleep(0.3)

    timeout = time.time() + 10.0
    while not piper.EnablePiper():
        if time.time() > timeout:
            print("FAILED")
            sys.exit(1)
        time.sleep(0.01)

    piper.GripperCtrl(0, 1000, 0x01, 0)
    time.sleep(0.5)
    print("Done")
    print()

    # Read current
    cur = read_joints_deg(piper)
    print("  Current joints:")
    for j in range(6):
        marker = " <<<" if j == joint_idx else ""
        print(f"    J{j+1}: {cur[j]:8.2f}°{marker}")
    print()

    # Test: step through positions
    # Generate test positions: 0, then steps toward min, back to 0, steps toward max, back to 0
    step = 15.0
    test_positions = [0.0]

    # Go toward minimum
    pos = 0.0
    while pos - step >= lo:
        pos -= step
        test_positions.append(pos)

    # Back to 0
    test_positions.append(0.0)

    # Go toward maximum
    pos = 0.0
    while pos + step <= hi:
        pos += step
        test_positions.append(pos)

    # Back to 0
    test_positions.append(0.0)

    print(f"  Will test {len(test_positions)} positions from {lo:.0f}° to {hi:.0f}°")
    print(f"  Speed: 50%")
    print()
    print(f"  {'Target':>10s}  {'Actual':>10s}  {'Error':>10s}  {'Status'}")
    print(f"  {'------':>10s}  {'------':>10s}  {'-----':>10s}  {'------'}")

    speed = 50
    failures = 0
    max_reached = 0.0
    min_reached = 0.0

    for target in test_positions:
        # Send command repeatedly for up to 5 seconds
        start = time.time()
        last_actual = read_joint_deg(piper, joint_idx)
        stall_start = None

        while time.time() - start < 5.0:
            send_joint_position(piper, joint_idx, target, speed)
            time.sleep(0.05)

            actual = read_joint_deg(piper, joint_idx)
            err = abs(target - actual)

            if err < 2.0:
                break

            # Detect stall: position not changing
            if abs(actual - last_actual) < 0.3:
                if stall_start is None:
                    stall_start = time.time()
                elif time.time() - stall_start > 1.5:
                    break  # Stalled for 1.5s
            else:
                stall_start = None
            last_actual = actual

        actual = read_joint_deg(piper, joint_idx)
        err = target - actual
        abs_err = abs(err)

        if actual > max_reached:
            max_reached = actual
        if actual < min_reached:
            min_reached = actual

        if abs_err < 2.0:
            status = "OK"
        elif abs_err < 10.0:
            status = "CLOSE"
        else:
            status = f"FAIL (stuck?)"
            failures += 1

        print(f"  {target:9.1f}°  {actual:9.1f}°  {err:+9.1f}°  {status}")

    # Return to 0
    for _ in range(50):
        send_joint_position(piper, joint_idx, 0.0, speed)
        time.sleep(0.05)

    print()
    print("=" * 60)
    print(f"  RESULTS:")
    print(f"    Range reached: {min_reached:.1f}° to {max_reached:.1f}°")
    print(f"    Expected:      {lo:.0f}° to {hi:.0f}°")
    print(f"    Failures:      {failures}/{len(test_positions)}")
    if failures == 0:
        print(f"    Verdict:       J{joint_num} WORKING")
    elif min_reached < lo * 0.5 or max_reached > hi * 0.5:
        print(f"    Verdict:       J{joint_num} PARTIAL — motor moves but limited range")
    else:
        print(f"    Verdict:       J{joint_num} FAULTY — motor not responding properly")
    print("=" * 60)

    piper.DisablePiper()


if __name__ == "__main__":
    main()
