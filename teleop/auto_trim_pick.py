#!/usr/bin/env python3
"""Auto-trim pick episodes to end right after gripper closes.

Reads a pick Zarr dataset, finds where the gripper first closes (>threshold)
in each episode, keeps a few hold frames after, and writes a new trimmed dataset.

Usage:
    python teleop/auto_trim_pick.py \
        --src ./data/split/pick \
        --dst ./data/split/pick_trimmed \
        --hold 10 --threshold 0.9
"""

import argparse
import json
import os
import sys

import numpy as np

try:
    import zarr
    from numcodecs import Blosc
except ImportError:
    print("zarr + numcodecs required: pip install zarr numcodecs")
    sys.exit(1)


def main():
    p = argparse.ArgumentParser(description="Auto-trim pick episodes")
    p.add_argument("--src", required=True, help="Source pick dataset")
    p.add_argument("--dst", required=True, help="Output trimmed dataset")
    p.add_argument("--hold", type=int, default=10,
                   help="Frames to keep after gripper closes (default: 10)")
    p.add_argument("--threshold", type=float, default=0.9,
                   help="Gripper close threshold (default: 0.9)")
    args = p.parse_args()

    # Load source
    src_zarr = os.path.join(args.src, "dataset.zarr")
    src = zarr.open_group(src_zarr, mode="r")
    src_data = src["data"]
    ends = np.array(src["meta"]["episode_ends"][:], dtype=np.int64)
    starts = np.concatenate([[0], ends[:-1]])
    n_eps = len(ends)

    # Load info
    info_path = os.path.join(args.src, "meta", "info.json")
    with open(info_path) as f:
        info = json.load(f)
    fps = info.get("fps", 20)
    cam_names = info.get("cameras", ["global", "wrist"])
    img_shape = info.get("image_shape", [720, 1280, 3])

    # Create destination
    dst_zarr = os.path.join(args.dst, "dataset.zarr")
    os.makedirs(args.dst, exist_ok=True)
    dst = zarr.open_group(dst_zarr, mode="w")
    data = dst.create_group("data")
    meta = dst.create_group("meta")

    comp = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    img_comp = Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE)

    data.create_dataset("state", shape=(0, 7), chunks=(256, 7),
                        dtype="float32", compressor=comp)
    data.create_dataset("action", shape=(0, 7), chunks=(256, 7),
                        dtype="float32", compressor=comp)
    data.create_dataset("eef_pos", shape=(0, 3), chunks=(256, 3),
                        dtype="float32", compressor=comp)
    data.create_dataset("eef_euler", shape=(0, 3), chunks=(256, 3),
                        dtype="float32", compressor=comp)
    data.create_dataset("timestamp", shape=(0,), chunks=(256,),
                        dtype="float32", compressor=comp)
    for cam in cam_names:
        key = f"img_{cam}"
        if key in src_data:
            data.create_dataset(key, shape=(0, img_shape[0], img_shape[1], 3),
                                chunks=(1, img_shape[0], img_shape[1], 3),
                                dtype="uint8", compressor=img_comp)
    meta.create_dataset("episode_ends", shape=(0,), chunks=(256,),
                        dtype="int64", compressor=comp)

    print(f"Source: {n_eps} episodes")
    print(f"Trimming to gripper close (>{args.threshold}) + {args.hold} hold frames\n")

    total_frames = 0
    ep_count = 0
    skipped = 0

    for i in range(n_eps):
        s, e = int(starts[i]), int(ends[i])
        grip = np.array(src_data["state"][s:e, 6])
        ep_len = e - s

        # Find first frame where gripper closes
        close_idx = np.argmax(grip > args.threshold)
        if grip[close_idx] <= args.threshold:
            print(f"  Ep {i}: gripper never closes, skipping")
            skipped += 1
            continue

        trim_end = min(close_idx + args.hold, ep_len)
        n = trim_end

        # Read trimmed data
        states = np.array(src_data["state"][s:s + n], dtype=np.float32)

        # Recompute actions with +1 shift
        actions = np.empty((n, 7), dtype=np.float32)
        for j in range(n - 1):
            actions[j] = states[j + 1]
        actions[n - 1] = states[n - 1]  # hold

        eef_pos = np.array(src_data["eef_pos"][s:s + n], dtype=np.float32)
        eef_euler = np.array(src_data["eef_euler"][s:s + n], dtype=np.float32)
        timestamps = np.arange(n, dtype=np.float32) / fps

        # Append
        data["state"].append(states)
        data["action"].append(actions)
        data["eef_pos"].append(eef_pos)
        data["eef_euler"].append(eef_euler)
        data["timestamp"].append(timestamps)

        for cam in cam_names:
            key = f"img_{cam}"
            if key in src_data and key in data:
                data[key].append(src_data[key][s:s + n])

        total_frames += n
        meta["episode_ends"].append(np.array([total_frames], dtype=np.int64))
        ep_count += 1

        print(f"  Ep {i}: {ep_len} -> {n} frames "
              f"(close at {close_idx}, {close_idx/fps:.1f}s)")

    # Write info
    os.makedirs(os.path.join(args.dst, "meta"), exist_ok=True)
    out_info = {
        "codebase_version": "zarr_v1",
        "robot_type": "piper_hanging",
        "fps": fps,
        "task": "pick object",
        "total_episodes": int(ep_count),
        "total_frames": int(total_frames),
        "cameras": cam_names,
        "image_shape": img_shape,
        "state_dim": 7,
        "action_dim": 7,
    }
    with open(os.path.join(args.dst, "meta", "info.json"), "w") as f:
        json.dump(out_info, f, indent=2)

    print(f"\nDone: {ep_count} episodes, {total_frames} frames "
          f"(skipped {skipped})")
    print(f"Output: {args.dst}")


if __name__ == "__main__":
    main()
