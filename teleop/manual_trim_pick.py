#!/usr/bin/env python3
"""Trim pick episodes using manually specified end frames."""

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


# Manual end frames per episode
END_FRAMES = {
    0: 351, 1: 308, 2: 338, 3: 236, 4: 271,
    5: 319, 6: 289, 7: 224, 8: 332, 9: 271,
    10: 266, 11: 282, 12: 214, 13: 226, 14: 205,
    15: 327, 16: 242, 17: 247, 18: 172, 19: 209,
    20: 167, 21: 197, 22: 189, 23: 142, 24: 210,
    25: 241, 26: 222, 27: 208, 28: 219, 29: 170,
    30: 206, 31: 203, 32: 160, 33: 132, 34: 176,
    35: 242, 36: 232, 37: 264, 38: 179, 39: 284,
    40: 152,
}

SRC = "./data/split/pick"
DST = "./data/split/pick_trimmed"


def main():
    src_zarr = os.path.join(SRC, "dataset.zarr")
    src = zarr.open_group(src_zarr, mode="r")
    src_data = src["data"]
    ends = np.array(src["meta"]["episode_ends"][:], dtype=np.int64)
    starts = np.concatenate([[0], ends[:-1]])
    n_eps = len(ends)

    with open(os.path.join(SRC, "meta", "info.json")) as f:
        info = json.load(f)
    fps = info.get("fps", 20)
    cam_names = info.get("cameras", ["global", "wrist"])
    img_shape = info.get("image_shape", [720, 1280, 3])

    # Create destination
    os.makedirs(DST, exist_ok=True)
    dst = zarr.open_group(os.path.join(DST, "dataset.zarr"), mode="w")
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

    total_frames = 0
    ep_count = 0

    for i in range(n_eps):
        if i not in END_FRAMES:
            print(f"  Ep {i}: no end frame specified, skipping")
            continue

        s = int(starts[i])
        ep_len = int(ends[i]) - s
        trim_end = min(END_FRAMES[i], ep_len)
        n = trim_end

        states = np.array(src_data["state"][s:s + n], dtype=np.float32)
        actions = np.empty((n, 7), dtype=np.float32)
        for j in range(n - 1):
            actions[j] = states[j + 1]
        actions[n - 1] = states[n - 1]

        eef_pos = np.array(src_data["eef_pos"][s:s + n], dtype=np.float32)
        eef_euler = np.array(src_data["eef_euler"][s:s + n], dtype=np.float32)
        timestamps = np.arange(n, dtype=np.float32) / fps

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

        grip_end = float(states[-1, 6])
        print(f"  Ep {i}: {ep_len} -> {n} frames ({n/fps:.1f}s) grip_end={grip_end:.2f}")

    os.makedirs(os.path.join(DST, "meta"), exist_ok=True)
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
    with open(os.path.join(DST, "meta", "info.json"), "w") as f:
        json.dump(out_info, f, indent=2)

    print(f"\nDone: {ep_count} episodes, {total_frames} frames")
    print(f"Output: {DST}")


if __name__ == "__main__":
    main()
