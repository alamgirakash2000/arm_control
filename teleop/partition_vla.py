#!/usr/bin/env python3
"""Partition full-task demos into labeled segments for VLA training.

You record ONE continuous demo (pick + place). This script splits it into
labeled segments and writes a single multi-task Zarr dataset.

Usage:
  # Step 1: Annotate split points
  python teleop/partition_vla.py --src ./data/demo_v2 --annotations vla_splits.json

  # Step 2: Export labeled dataset
  python teleop/partition_vla.py --src ./data/demo_v2 --annotations vla_splits.json \
      --export ./data/vla_dataset

Controls during annotation:
  Space       — play / pause
  Right / D   — jump forward 10 frames
  Left / A    — jump backward 10 frames
  1           — mark end of PICK (start of PLACE)
  S           — save and next episode
  Q           — quit (progress saved)
  R           — reset markers
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

try:
    import zarr
    from numcodecs import Blosc
except ImportError:
    print("zarr + numcodecs required: pip install zarr numcodecs")
    sys.exit(1)


# ── Default task labels — customize these for your tasks ──────────────────
PICK_LABEL = "pick the object from the table"
PLACE_LABEL = "place the object on the good box"


def load_zarr_dataset(src_dir):
    zarr_path = os.path.join(src_dir, "dataset.zarr")
    root = zarr.open_group(zarr_path, mode="r")
    episode_ends = np.array(root["meta"]["episode_ends"][:], dtype=np.int64)
    info_path = os.path.join(src_dir, "meta", "info.json")
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
    else:
        info = {"fps": 20, "total_episodes": len(episode_ends),
                "total_frames": int(episode_ends[-1]) if len(episode_ends) > 0 else 0}
    episodes = []
    ep_start = 0
    for i, ep_end in enumerate(episode_ends):
        ep_end = int(ep_end)
        episodes.append({"episode_index": i, "start": ep_start,
                         "end": ep_end, "length": ep_end - ep_start})
        ep_start = ep_end
    return root, info, episodes


def annotate_episode(ep_idx, frames, fps):
    n = len(frames)
    if n == 0:
        return "skip"
    pick_end = None
    pos = 0
    playing = False
    step_size = 10

    win = f"Episode {ep_idx} — Mark PICK end"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 960, 540)

    while True:
        frame = frames[pos].copy()
        h, w = frame.shape[:2]

        # Timeline
        bar_y = h - 40
        cv2.rectangle(frame, (0, bar_y), (w, bar_y + 20), (40, 40, 40), -1)
        px = int(pos / max(n - 1, 1) * w)
        cv2.rectangle(frame, (0, bar_y), (px, bar_y + 20), (100, 100, 100), -1)
        if pick_end is not None:
            mx = int(pick_end / max(n - 1, 1) * w)
            cv2.line(frame, (mx, bar_y - 5), (mx, bar_y + 25), (0, 255, 0), 2)
            cv2.putText(frame, "PICK|PLACE", (mx - 30, bar_y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.circle(frame, (px, bar_y + 10), 6, (255, 255, 255), -1)

        # Info
        phase = "?"
        if pick_end is not None:
            phase = "PICK" if pos <= pick_end else "PLACE"
        lines = [
            f"Frame: {pos}/{n-1} ({pos/fps:.1f}s)  Phase: {phase}",
            f"PICK end: {'frame ' + str(pick_end) if pick_end else '(press 1)'}",
            f"Space=play  Arrows=jump  1=mark  S=save  Q=quit",
        ]
        for i, txt in enumerate(lines):
            cv2.putText(frame, txt, (10, 25 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        cv2.imshow(win, frame)
        wait_ms = 50 if playing else 30
        key = cv2.waitKeyEx(wait_ms)

        if playing and key == -1:
            pos = min(pos + 1, n - 1)
            if pos >= n - 1:
                playing = False
            continue

        k = key & 0xFF
        if k == ord(' '):
            playing = not playing
        elif k == ord('q') or k == ord('Q'):
            cv2.destroyWindow(win)
            return None
        elif key == 65363 or k == ord('d'):
            pos = min(pos + step_size, n - 1)
            playing = False
        elif key == 65361 or k == ord('a'):
            pos = max(pos - step_size, 0)
            playing = False
        elif k == ord('1'):
            pick_end = pos
            print(f"  PICK end = frame {pos} ({pos/fps:.1f}s)")
        elif k == ord('r'):
            pick_end = None
        elif k == ord('s'):
            if pick_end is None:
                print("  Set PICK end first (press 1)")
                continue
            cv2.destroyWindow(win)
            return {"pick_end": pick_end}


def export_vla_dataset(src_root, src_info, annotations, episodes, out_dir,
                       pick_label, place_label):
    """Export a single multi-task Zarr dataset with text labels per frame."""
    zarr_path = os.path.join(out_dir, "dataset.zarr")
    os.makedirs(out_dir, exist_ok=True)
    dst = zarr.open_group(zarr_path, mode="w")
    data = dst.create_group("data")
    meta = dst.create_group("meta")

    fps = src_info.get("fps", 20)
    cam_names = src_info.get("cameras", ["global", "wrist"])
    img_shape = src_info.get("image_shape", [720, 1280, 3])
    img_h, img_w = img_shape[0], img_shape[1]

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
        if key in src_root["data"]:
            data.create_dataset(key, shape=(0, img_h, img_w, 3),
                                chunks=(1, img_h, img_w, 3),
                                dtype="uint8", compressor=img_comp)
    meta.create_dataset("episode_ends", shape=(0,), chunks=(256,),
                        dtype="int64", compressor=comp)

    src_data = src_root["data"]
    total_frames = 0
    ep_count = 0
    task_labels = []  # per-episode: {"episode_index", "task", "start", "end"}

    for ann in annotations:
        ep_idx = ann["episode_index"]
        ep = next(e for e in episodes if e["episode_index"] == ep_idx)
        src_start = ep["start"]
        ep_len = ep["length"]
        pick_end = ann["pick_end"]

        # Segment 1: PICK (frame 0 to pick_end inclusive)
        for seg_name, seg_label, seg_start, seg_end in [
            ("pick", pick_label, 0, pick_end + 1),
            ("place", place_label, pick_end + 1, ep_len),
        ]:
            abs_start = src_start + seg_start
            abs_end = src_start + seg_end
            n = abs_end - abs_start
            if n < 16:  # skip too-short segments
                print(f"  Ep {ep_idx} {seg_name}: too short ({n} frames), skipping")
                continue

            states = np.array(src_data["state"][abs_start:abs_end], dtype=np.float32)
            # Recompute actions
            actions = np.empty((n, 7), dtype=np.float32)
            for i in range(n - 1):
                actions[i] = states[i + 1]
            actions[n - 1] = states[n - 1]

            eef_pos = np.array(src_data["eef_pos"][abs_start:abs_end], dtype=np.float32)
            eef_euler = np.array(src_data["eef_euler"][abs_start:abs_end], dtype=np.float32)
            timestamps = np.arange(n, dtype=np.float32) / fps

            data["state"].append(states)
            data["action"].append(actions)
            data["eef_pos"].append(eef_pos)
            data["eef_euler"].append(eef_euler)
            data["timestamp"].append(timestamps)

            for cam in cam_names:
                key = f"img_{cam}"
                if key in src_data and key in data:
                    data[key].append(src_data[key][abs_start:abs_end])

            total_frames += n
            meta["episode_ends"].append(np.array([total_frames], dtype=np.int64))

            task_labels.append({
                "episode_index": ep_count,
                "task": seg_label,
                "source_episode": ep_idx,
                "segment": seg_name,
                "frames": n,
            })
            ep_count += 1
            print(f"  Ep {ep_idx} {seg_name}: {n} frames → \"{seg_label}\"")

    # Write metadata
    os.makedirs(os.path.join(out_dir, "meta"), exist_ok=True)
    info = {
        "codebase_version": "zarr_v1",
        "robot_type": "piper_hanging",
        "fps": fps,
        "total_episodes": int(ep_count),
        "total_frames": int(total_frames),
        "cameras": cam_names,
        "image_shape": img_shape,
        "state_dim": 7,
        "action_dim": 7,
    }
    with open(os.path.join(out_dir, "meta", "info.json"), "w") as f:
        json.dump(info, f, indent=2)

    # Write task labels — this is the key file for VLA training
    with open(os.path.join(out_dir, "meta", "task_labels.json"), "w") as f:
        json.dump(task_labels, f, indent=2)

    # Write unique tasks list
    tasks = sorted(set(t["task"] for t in task_labels))
    with open(os.path.join(out_dir, "meta", "tasks.json"), "w") as f:
        json.dump(tasks, f, indent=2)

    print(f"\n  Done: {ep_count} segments, {total_frames} frames")
    print(f"  Tasks: {tasks}")
    print(f"  Output: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Partition demos for VLA training")
    parser.add_argument("--src", required=True, help="Source dataset dir")
    parser.add_argument("--annotations", default="vla_splits.json")
    parser.add_argument("--export", default=None, help="Export directory")
    parser.add_argument("--cam", default="global", help="Camera for annotation display")
    parser.add_argument("--pick-label", default=PICK_LABEL)
    parser.add_argument("--place-label", default=PLACE_LABEL)
    args = parser.parse_args()

    src_dir = os.path.abspath(args.src)
    root, info, episodes = load_zarr_dataset(src_dir)
    fps = info.get("fps", 20)
    print(f"Source: {src_dir}")
    print(f"  {len(episodes)} episodes, {info.get('total_frames', 0)} frames")

    ann_path = os.path.abspath(args.annotations)
    saved = {}
    if os.path.exists(ann_path):
        with open(ann_path) as f:
            for a in json.load(f):
                saved[a["episode_index"]] = a
        print(f"  Loaded {len(saved)} annotations")

    if args.export:
        if not saved:
            print("No annotations! Run without --export first.")
            sys.exit(1)
        ann_list = []
        for ep in episodes:
            if ep["episode_index"] in saved:
                ann_list.append({**ep, **saved[ep["episode_index"]]})
        ann_list.sort(key=lambda a: a["episode_index"])
        print(f"\nExporting {len(ann_list)} episodes...")
        export_vla_dataset(root, info, ann_list, episodes, args.export,
                           args.pick_label, args.place_label)
        return

    # Annotation mode
    print(f"\nAnnotating... ({len(saved)} done)")
    print(f"  PICK label: \"{args.pick_label}\"")
    print(f"  PLACE label: \"{args.place_label}\"\n")

    for ep in episodes:
        idx = ep["episode_index"]
        if idx in saved:
            print(f"  Ep {idx}: already annotated (pick_end={saved[idx]['pick_end']})")
            continue

        print(f"\n  Ep {idx} ({ep['length']} frames, {ep['length']/fps:.1f}s):")
        frames = root["data"][f"img_{args.cam}"][ep["start"]:ep["end"]]
        if len(frames) == 0:
            continue

        result = annotate_episode(idx, frames, fps)
        if result is None:
            break
        elif result == "skip":
            continue
        else:
            saved[idx] = {"episode_index": idx, "length": ep["length"],
                          "pick_end": result["pick_end"]}

    if saved:
        with open(ann_path, "w") as f:
            json.dump(sorted(saved.values(), key=lambda a: a["episode_index"]), f, indent=2)
        print(f"\nSaved {len(saved)} annotations to {ann_path}")


if __name__ == "__main__":
    main()
