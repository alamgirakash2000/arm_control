#!/usr/bin/env python3
"""Partition full-task Zarr episodes into sub-task datasets.

Usage:
  # Step 1: Annotate split points interactively
  python partition_episodes.py --src ./data/demo --annotations splits.json

  # Step 2: Export sub-datasets from saved annotations
  python partition_episodes.py --src ./data/demo --annotations splits.json --export ./data

Controls during annotation:
  Space                — play / pause
  Right arrow / D      — jump forward 0.5s (10 frames)
  Left arrow / A       — jump backward 0.5s (10 frames)
  1                    — mark end of PICK (start of PLACE)
  S                    — save annotations and move to next episode
  Q                    — quit (annotations saved so far are kept)
  R                    — reset markers for current episode
  +/-                  — speed up / slow down playback
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


def load_zarr_dataset(src_dir):
    """Load Zarr dataset metadata and return root group + episode info."""
    zarr_path = os.path.join(src_dir, "dataset.zarr")
    root = zarr.open_group(zarr_path, mode="r")
    episode_ends = np.array(root["meta"]["episode_ends"][:], dtype=np.int64)

    info_path = os.path.join(src_dir, "meta", "info.json")
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
    else:
        info = {
            "fps": 20,
            "total_episodes": len(episode_ends),
            "total_frames": int(episode_ends[-1]) if len(episode_ends) > 0 else 0,
        }

    # Build episode list
    episodes = []
    ep_start = 0
    for i, ep_end in enumerate(episode_ends):
        ep_end = int(ep_end)
        episodes.append({
            "episode_index": i,
            "start": ep_start,
            "end": ep_end,
            "length": ep_end - ep_start,
        })
        ep_start = ep_end

    return root, info, episodes


def load_video_frames_zarr(root, ep_start, ep_end, cam_name="global"):
    """Load video frames from Zarr img array for an episode."""
    key = f"img_{cam_name}"
    if key not in root["data"]:
        return []
    return root["data"][key][ep_start:ep_end]


def annotate_episode(ep_idx, frames, total_frames, fps, existing_marks=None):
    """Interactive annotation of a single episode.

    Returns dict with 'pick_end' frame index, or None if user quits.
    """
    n = len(frames)
    if n == 0:
        print(f"  No video frames for episode {ep_idx}, skipping")
        return "skip"

    pick_end = existing_marks.get("pick_end") if existing_marks else None
    pos = 0
    playing = False
    step_size = 10  # frames per arrow press (0.5s at 20fps)
    speed = 1
    quit_flag = False

    win_name = f"Episode {ep_idx} — Annotate Split Points"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 960, 540)

    def draw():
        frame = frames[pos].copy()
        h, w = frame.shape[:2]

        # Draw timeline bar at bottom
        bar_y = h - 40
        bar_h = 20
        cv2.rectangle(frame, (0, bar_y), (w, bar_y + bar_h), (40, 40, 40), -1)

        # Progress indicator
        px = int(pos / max(n - 1, 1) * w)
        cv2.rectangle(frame, (0, bar_y), (px, bar_y + bar_h), (100, 100, 100), -1)

        # Draw split marker on timeline
        if pick_end is not None:
            mx = int(pick_end / max(n - 1, 1) * w)
            cv2.line(frame, (mx, bar_y - 5), (mx, bar_y + bar_h + 5),
                     (0, 255, 0), 2)
            cv2.putText(frame, "PICK|PLACE", (mx - 30, bar_y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Current position marker
        cv2.circle(frame, (px, bar_y + bar_h // 2), 6, (255, 255, 255), -1)

        # Info overlay
        info_lines = [
            f"Frame: {pos}/{n-1}  ({pos/fps:.1f}s / {(n-1)/fps:.1f}s)",
            f"PICK end: {'frame ' + str(pick_end) if pick_end is not None else '(not set)  [press 1]'}",
            f"Step: {step_size}f ({step_size/fps:.1f}s)  |  "
            f"Space=play  Arrows=jump  S=save  R=reset  Q=quit",
        ]
        for i, txt in enumerate(info_lines):
            cv2.putText(frame, txt, (10, 25 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                        cv2.LINE_AA)

        # Phase label
        if pick_end is not None:
            if pos <= pick_end:
                phase, color = "PICK", (0, 255, 0)
            else:
                phase, color = "PLACE", (0, 100, 255)
            cv2.putText(frame, phase, (w - 150, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

        cv2.imshow(win_name, frame)

    KEY_LEFT = 65361
    KEY_RIGHT = 65363

    while True:
        draw()
        wait_ms = 50 if playing else 30
        key = cv2.waitKeyEx(wait_ms)

        if playing and key == -1:
            pos = min(pos + speed, n - 1)
            if pos >= n - 1:
                playing = False
            continue

        key_char = key & 0xFF

        if key_char == ord(' '):
            playing = not playing
        elif key_char == ord('q') or key_char == ord('Q'):
            quit_flag = True
            break
        elif key == KEY_RIGHT or key_char == ord('d'):
            pos = min(pos + step_size, n - 1)
            playing = False
        elif key == KEY_LEFT or key_char == ord('a'):
            pos = max(pos - step_size, 0)
            playing = False
        elif key_char == ord('1'):
            pick_end = pos
            print(f"  PICK end = frame {pos} ({pos/fps:.1f}s)")
        elif key_char == ord('r') or key_char == ord('R'):
            pick_end = None
            print("  Markers reset")
        elif key_char == ord('s') or key_char == ord('S'):
            if pick_end is None:
                print("  PICK end must be set before saving! (press 1)")
                continue
            break
        elif key_char == ord('+') or key_char == ord('='):
            step_size = min(step_size + 5, 40)
            print(f"  Step: {step_size} frames ({step_size/fps:.1f}s)")
        elif key_char == ord('-'):
            step_size = max(step_size - 5, 1)
            print(f"  Step: {step_size} frames ({step_size/fps:.1f}s)")

    cv2.destroyWindow(win_name)

    if quit_flag:
        return None
    return {"pick_end": pick_end}


def export_subtask_zarr(src_root, src_info, annotations, task_name,
                        out_dir, frame_selector, task_label):
    """Export a sub-task Zarr dataset from annotated episodes."""
    zarr_path = os.path.join(out_dir, "dataset.zarr")
    dst = zarr.open_group(zarr_path, mode="w")
    data = dst.create_group("data")
    meta = dst.create_group("meta")

    fps = src_info.get("fps", 20)
    cam_names = src_info.get("cameras", ["global", "wrist"])
    img_shape = src_info.get("image_shape", [720, 1280, 3])
    img_h, img_w = img_shape[0], img_shape[1]

    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    img_compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE)

    # Create arrays
    data.create_dataset("state", shape=(0, 7), chunks=(256, 7),
                        dtype="float32", compressor=compressor)
    data.create_dataset("action", shape=(0, 7), chunks=(256, 7),
                        dtype="float32", compressor=compressor)
    data.create_dataset("eef_pos", shape=(0, 3), chunks=(256, 3),
                        dtype="float32", compressor=compressor)
    data.create_dataset("eef_euler", shape=(0, 3), chunks=(256, 3),
                        dtype="float32", compressor=compressor)
    data.create_dataset("timestamp", shape=(0,), chunks=(256,),
                        dtype="float32", compressor=compressor)
    for cam in cam_names:
        key = f"img_{cam}"
        if key in src_root["data"]:
            data.create_dataset(key, shape=(0, img_h, img_w, 3),
                                chunks=(1, img_h, img_w, 3),
                                dtype="uint8", compressor=img_compressor)
    meta.create_dataset("episode_ends", shape=(0,), chunks=(256,),
                        dtype="int64", compressor=compressor)

    src_data = src_root["data"]
    total_frames = 0
    ep_count = 0

    for ann in annotations:
        src_start = ann["start"]
        src_len = ann["length"]
        sel_start, sel_end = frame_selector(ann)

        if sel_start >= sel_end:
            print(f"  Skipping episode {ann['episode_index']} for {task_name}: "
                  f"empty range [{sel_start}, {sel_end})")
            continue

        abs_start = src_start + sel_start
        abs_end = src_start + sel_end
        n = abs_end - abs_start

        # Slice source arrays
        states = np.array(src_data["state"][abs_start:abs_end], dtype=np.float32)

        # Recompute actions with +1 shift for the sub-episode
        actions = np.empty((n, 7), dtype=np.float32)
        for i in range(n - 1):
            actions[i] = states[i + 1]
        actions[n - 1] = states[n - 1]  # hold position

        eef_pos = np.array(src_data["eef_pos"][abs_start:abs_end], dtype=np.float32)
        eef_euler = np.array(src_data["eef_euler"][abs_start:abs_end], dtype=np.float32)

        # Recompute timestamps relative to sub-episode start
        timestamps = np.arange(n, dtype=np.float32) / fps

        # Append to destination
        data["state"].append(states)
        data["action"].append(actions)
        data["eef_pos"].append(eef_pos)
        data["eef_euler"].append(eef_euler)
        data["timestamp"].append(timestamps)

        for cam in cam_names:
            key = f"img_{cam}"
            if key in src_data and key in data:
                img_chunk = src_data[key][abs_start:abs_end]
                data[key].append(img_chunk)

        total_frames += n
        meta["episode_ends"].append(np.array([total_frames], dtype=np.int64))
        ep_count += 1

        print(f"  {task_name} ep {ep_count - 1}: "
              f"frames {sel_start}-{sel_end - 1} from src ep {ann['episode_index']} "
              f"({n} frames, {n / fps:.1f}s)")

    # Write metadata JSON
    os.makedirs(os.path.join(out_dir, "meta"), exist_ok=True)
    info = {
        "codebase_version": "zarr_v1",
        "robot_type": "piper_hanging",
        "fps": fps,
        "task": task_label,
        "total_episodes": ep_count,
        "total_frames": total_frames,
        "cameras": cam_names,
        "image_shape": img_shape,
        "state_dim": 7,
        "action_dim": 7,
    }
    with open(os.path.join(out_dir, "meta", "info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print(f"  {task_name}: {ep_count} episodes, {total_frames} frames total")


def main():
    parser = argparse.ArgumentParser(
        description="Partition full-task Zarr episodes into sub-task datasets")
    parser.add_argument("--src", required=True,
                        help="Source Zarr dataset directory (e.g. ./data/demo)")
    parser.add_argument("--annotations", default="splits.json",
                        help="JSON file to save/load split annotations")
    parser.add_argument("--export", default=None,
                        help="Export sub-datasets to this directory "
                             "(creates pick/, place_good/ or place_bad/ inside)")
    parser.add_argument("--cam", default="global",
                        help="Camera name for annotation display (default: global)")
    parser.add_argument("--task-type", default="good",
                        choices=["good", "bad"],
                        help="Whether these demos are good or bad "
                             "(affects place sub-task naming)")
    args = parser.parse_args()

    src_dir = os.path.abspath(args.src)
    root, info, episodes = load_zarr_dataset(src_dir)
    n_eps = len(episodes)
    fps = info.get("fps", 20)
    print(f"Source dataset: {src_dir}")
    print(f"  {n_eps} episodes, {info.get('total_frames', 0)} total frames, {fps} fps")

    # Load existing annotations
    ann_path = os.path.abspath(args.annotations)
    saved = {}
    if os.path.exists(ann_path):
        with open(ann_path) as f:
            saved_list = json.load(f)
        for a in saved_list:
            saved[a["episode_index"]] = a
        print(f"  Loaded {len(saved)} existing annotations from {ann_path}")

    if args.export:
        # Export mode
        if not saved:
            print("No annotations found! Run without --export first.")
            sys.exit(1)

        # Merge annotation data with episode metadata
        ann_list = []
        for ep in episodes:
            ep_idx = ep["episode_index"]
            if ep_idx in saved:
                merged = {**ep, **saved[ep_idx]}
                ann_list.append(merged)
        ann_list.sort(key=lambda a: a["episode_index"])
        print(f"\nExporting {len(ann_list)} annotated episodes...")

        place_name = "place_good" if args.task_type == "good" else "place_bad"

        # PICK: frame 0 to pick_end (inclusive)
        pick_dir = os.path.join(args.export, "pick")
        print(f"\n--- PICK → {pick_dir} ---")
        export_subtask_zarr(
            root, info, ann_list, "PICK", pick_dir,
            frame_selector=lambda a: (0, a["pick_end"] + 1),
            task_label="pick object")

        # PLACE: pick_end+1 to end
        place_dir = os.path.join(args.export, place_name)
        print(f"\n--- {place_name.upper()} → {place_dir} ---")
        export_subtask_zarr(
            root, info, ann_list, place_name.upper(), place_dir,
            frame_selector=lambda a: (a["pick_end"] + 1, a["length"]),
            task_label=f"place object {args.task_type}")

        print("\nExport complete!")
        return

    # Annotation mode
    print(f"\nAnnotating episodes... ({len(saved)} already done)")
    print("Controls: Space=play, Arrows=jump, 1=PICK end, S=save, R=reset, Q=quit\n")

    for ep in episodes:
        ep_idx = ep["episode_index"]
        ep_len = ep["length"]
        ep_start = ep["start"]
        ep_end = ep["end"]

        if ep_idx in saved:
            s = saved[ep_idx]
            print(f"Episode {ep_idx} ({ep_len} frames): "
                  f"already annotated — PICK end={s['pick_end']}")
            continue

        print(f"\nEpisode {ep_idx} ({ep_len} frames, {ep_len / fps:.1f}s):")
        frames = load_video_frames_zarr(root, ep_start, ep_end, args.cam)
        if len(frames) == 0:
            print("  No video frames, skipping")
            continue

        result = annotate_episode(ep_idx, frames, ep_len, fps)

        if result is None:
            print("\nQuitting. Saving annotations so far...")
            break
        elif result == "skip":
            continue
        else:
            saved[ep_idx] = {
                "episode_index": ep_idx,
                "length": ep_len,
                "pick_end": result["pick_end"],
            }
            print(f"  Saved: PICK [0-{result['pick_end']}], "
                  f"PLACE [{result['pick_end'] + 1}-{ep_len - 1}]")

    # Save annotations
    if saved:
        ann_list = sorted(saved.values(), key=lambda a: a["episode_index"])
        with open(ann_path, "w") as f:
            json.dump(ann_list, f, indent=2)
        print(f"\nSaved {len(ann_list)} annotations to {ann_path}")
    else:
        print("\nNo annotations to save.")


if __name__ == "__main__":
    main()
