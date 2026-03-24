#!/usr/bin/env python3
"""
Gripper diagnostics — run this to test two things:
  1. Direct gripper hardware control (open/close)
  2. What gripper values are stored in a dataset episode

Usage:
    # Test hardware only:
    python3 teleop/test_gripper.py --can can0

    # Test hardware + inspect dataset:
    python3 teleop/test_gripper.py --can can0 --dataset data/tool_good --episode 0

    # Inspect dataset only (no robot):
    python3 teleop/test_gripper.py --dataset data/tool_good --episode 0
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_hardware(can_port):
    import logging
    logging.basicConfig(level=logging.WARNING)

    from piper_core import PiperHangingController
    print(f"\n=== Hardware gripper test (CAN: {can_port}) ===")
    ctrl = PiperHangingController(can_port=can_port)
    if not ctrl.connect():
        print("ERROR: Failed to connect to robot")
        return

    piper = ctrl.piper

    def print_gripper_status(label):
        try:
            g = piper.GetArmGripperMsgs()
            gs = g.gripper_state
            print(f"  [{label}] angle={gs.grippers_angle} effort={gs.grippers_effort} "
                  f"foc_status={gs.foc_status}")
        except Exception as e:
            print(f"  [{label}] read error: {e}")

    print("  Reading gripper status before commands:")
    print_gripper_status("initial")

    # Try enable + clear error (0x03) which is more aggressive than 0x01
    print("  Sending 0x03 (enable+clear error) init ...")
    for _ in range(20):
        piper.GripperCtrl(0, 1000, 0x03, 0)
        time.sleep(0.005)
    time.sleep(0.3)
    print_gripper_status("after 0x03 init")

    steps = [0, 20, 40, 70, 40, 20, 0]
    for mm in steps:
        um = mm * 1000
        print(f"  Sending {mm} mm ({um} um) ...", flush=True)
        for _ in range(40):  # send at 200Hz for 200ms
            piper.GripperCtrl(um, 1000, 0x01, 0)
            time.sleep(0.005)
        time.sleep(1.0)
        print_gripper_status(f"after {mm}mm")

    print("  Hardware test done.")
    ctrl.shutdown()


def inspect_dataset(dataset_dir, episode_idx):
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("ERROR: pyarrow not installed (pip install pyarrow)")
        return

    parquet_path = os.path.join(
        dataset_dir, "data", "chunk-000",
        f"episode_{episode_idx:06d}.parquet")

    if not os.path.exists(parquet_path):
        print(f"ERROR: {parquet_path} not found")
        return

    print(f"\n=== Dataset inspection: {parquet_path} ===")
    table = pq.read_table(parquet_path)
    data = table.to_pydict()

    print(f"  Columns: {list(data.keys())}")
    n = len(data["timestamp"])
    print(f"  Frames: {n}")

    # Check action.abs_joint for gripper column (index 6)
    if "action.abs_joint" in data:
        actions = data["action.abs_joint"]
        grip_values = []
        for i, a in enumerate(actions):
            a = list(a)
            if len(a) > 6:
                grip_values.append((i, float(a[6])))

        if grip_values:
            grips = [v for _, v in grip_values]
            print(f"\n  action.abs_joint[6] (gripper):")
            print(f"    min={min(grips):.4f}  max={max(grips):.4f}")
            print(f"    First 5: {[f'{v:.4f}' for _, v in grip_values[:5]]}")
            print(f"    Last  5: {[f'{v:.4f}' for _, v in grip_values[-5:]]}")

            # Find where gripper first goes nonzero
            nonzero = [(i, v) for i, v in grip_values if v > 0.01]
            if nonzero:
                fi, fv = nonzero[0]
                print(f"    First nonzero: frame {fi} = {fv:.4f}")
                # Show around that frame
                print(f"    Frames {fi-2}..{fi+3}:")
                for i, v in grip_values[max(0,fi-2):fi+4]:
                    print(f"      [{i:4d}] {v:.4f}")
            else:
                print(f"    WARNING: gripper is ALWAYS ZERO in this episode!")
        else:
            print("  action.abs_joint has no index [6] — only 6 joints stored!")
    else:
        print("  action.abs_joint not found in dataset!")

    # Also check observation.state
    if "observation.state" in data:
        states = data["observation.state"]
        grip_obs = [(i, float(list(s)[6])) for i, s in enumerate(states) if len(list(s)) > 6]
        if grip_obs:
            grips = [v for _, v in grip_obs]
            print(f"\n  observation.state[6] (gripper):")
            print(f"    min={min(grips):.4f}  max={max(grips):.4f}")
            nonzero = [(i, v) for i, v in grip_obs if v > 0.01]
            if nonzero:
                print(f"    First nonzero: frame {nonzero[0][0]} = {nonzero[0][1]:.4f}")
            else:
                print(f"    WARNING: gripper observation is ALWAYS ZERO!")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--can", default=None, help="CAN port (e.g. can0)")
    p.add_argument("--dataset", default=None)
    p.add_argument("--episode", type=int, default=0)
    args = p.parse_args()

    if args.can:
        test_hardware(args.can)

    if args.dataset:
        inspect_dataset(args.dataset, args.episode)

    if not args.can and not args.dataset:
        print("Usage: python3 test_gripper.py --can can0 [--dataset path --episode 0]")


if __name__ == "__main__":
    main()
