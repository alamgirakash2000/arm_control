#!/usr/bin/env python3
"""Partition full-task episodes into sub-task datasets.

Usage:
  # Step 1: Annotate split points interactively
  python partition_episodes.py --src ./data/tool_good --annotations splits.json

  # Step 2: Export sub-datasets from saved annotations
  python partition_episodes.py --src ./data/tool_good --annotations splits.json --export ./data

Controls during annotation:
  Space                — play / pause
  Right arrow / D      — jump forward 0.5s (10 frames)
  Left arrow / A       — jump backward 0.5s (10 frames)
  1                    — mark end of PICK (start of INSPECT)
  2                    — mark end of INSPECT (start of PLACE)
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
    import pyarrow.parquet as pq
except ImportError:
    print("pyarrow required: pip install pyarrow")
    sys.exit(1)


def load_episode_info(src_dir):
    """Load episode metadata from the source dataset."""
    episodes_path = os.path.join(src_dir, "meta", "episodes.jsonl")
    info_path = os.path.join(src_dir, "meta", "info.json")

    with open(info_path) as f:
        info = json.load(f)

    episodes = []
    with open(episodes_path) as f:
        for line in f:
            episodes.append(json.loads(line.strip()))

    return info, episodes


def load_parquet(src_dir, ep_idx):
    """Load a parquet file for an episode."""
    path = os.path.join(src_dir, "data", "chunk-000",
                        f"episode_{ep_idx:06d}.parquet")
    return pq.read_table(path).to_pandas()


def load_video_frames(src_dir, ep_idx, cam_name="cam_0"):
    """Load all frames from an episode video."""
    vpath = os.path.join(src_dir, "videos", "chunk-000",
                         f"observation.images.{cam_name}",
                         f"episode_{ep_idx:06d}.mp4")
    if not os.path.exists(vpath):
        return []

    cap = cv2.VideoCapture(vpath)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def annotate_episode(ep_idx, frames, total_frames, existing_marks=None):
    """Interactive annotation of a single episode.

    Returns dict with 'pick_end' and 'inspect_end' frame indices,
    or None if user quits.
    """
    n = len(frames)
    if n == 0:
        print(f"  No video frames for episode {ep_idx}, skipping")
        return "skip"

    pick_end = existing_marks.get("pick_end") if existing_marks else None
    inspect_end = existing_marks.get("inspect_end") if existing_marks else None
    pos = 0
    playing = False
    step_size = 10  # frames per arrow press (0.5s at 20fps)
    speed = 1  # playback speed multiplier
    quit_flag = False

    win_name = f"Episode {ep_idx} — Annotate Split Points"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 960, 720)

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

        # Draw split markers on timeline
        if pick_end is not None:
            mx = int(pick_end / max(n - 1, 1) * w)
            cv2.line(frame, (mx, bar_y - 5), (mx, bar_y + bar_h + 5),
                     (0, 255, 0), 2)
            cv2.putText(frame, "1", (mx - 5, bar_y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if inspect_end is not None:
            mx = int(inspect_end / max(n - 1, 1) * w)
            cv2.line(frame, (mx, bar_y - 5), (mx, bar_y + bar_h + 5),
                     (0, 200, 255), 2)
            cv2.putText(frame, "2", (mx - 5, bar_y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        # Current position marker
        cv2.circle(frame, (px, bar_y + bar_h // 2), 6, (255, 255, 255), -1)

        # Info overlay
        info_lines = [
            f"Frame: {pos}/{n-1}  ({pos/20:.1f}s / {(n-1)/20:.1f}s)",
            f"PICK end:    {'frame ' + str(pick_end) if pick_end is not None else '(not set)  [press 1]'}",
            f"INSPECT end: {'frame ' + str(inspect_end) if inspect_end is not None else '(not set)  [press 2]'}",
            f"Step: {step_size}f ({step_size/20:.1f}s)  |  Space=play  Arrows=jump  S=save  R=reset  Q=quit",
        ]
        for i, txt in enumerate(info_lines):
            cv2.putText(frame, txt, (10, 25 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                        cv2.LINE_AA)

        # Phase label
        if pick_end is not None and inspect_end is not None:
            if pos <= pick_end:
                phase = "PICK"
                color = (0, 255, 0)
            elif pos <= inspect_end:
                phase = "INSPECT"
                color = (0, 200, 255)
            else:
                phase = "PLACE"
                color = (0, 100, 255)
            cv2.putText(frame, phase, (w - 150, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

        cv2.imshow(win_name, frame)

    # Arrow key codes for cv2.waitKeyEx (platform-dependent)
    KEY_LEFT = 65361
    KEY_RIGHT = 65363

    while True:
        draw()
        wait_ms = 50 if playing else 30
        key = cv2.waitKeyEx(wait_ms)

        if playing and key == -1:  # no key pressed during playback
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
            print(f"  PICK end = frame {pos} ({pos/20:.1f}s)")
        elif key_char == ord('2'):
            inspect_end = pos
            print(f"  INSPECT end = frame {pos} ({pos/20:.1f}s)")
        elif key_char == ord('r') or key_char == ord('R'):
            pick_end = None
            inspect_end = None
            print("  Markers reset")
        elif key_char == ord('s') or key_char == ord('S'):
            if pick_end is None or inspect_end is None:
                print("  Both markers must be set before saving! (press 1 and 2)")
                continue
            if pick_end >= inspect_end:
                print("  PICK end must be before INSPECT end!")
                continue
            break
        elif key_char == ord('+') or key_char == ord('='):
            step_size = min(step_size + 5, 40)
            print(f"  Step: {step_size} frames ({step_size/20:.1f}s)")
        elif key_char == ord('-'):
            step_size = max(step_size - 5, 1)
            print(f"  Step: {step_size} frames ({step_size/20:.1f}s)")

    cv2.destroyWindow(win_name)

    if quit_flag:
        return None

    return {"pick_end": pick_end, "inspect_end": inspect_end}


def export_subtask(src_dir, src_info, annotations, task_name, out_dir,
                   frame_selector, task_label, cam_name="cam_0"):
    """Export a sub-task dataset from annotated episodes.

    frame_selector: function(annotation, ep_length) -> (start_frame, end_frame)
    """
    os.makedirs(os.path.join(out_dir, "data", "chunk-000"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "meta"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "videos", "chunk-000",
                             f"observation.images.{cam_name}"), exist_ok=True)

    fps = src_info["fps"]
    total_frames = 0
    ep_count = 0

    for ann in annotations:
        ep_idx = ann["episode_index"]
        src_len = ann["length"]
        start, end = frame_selector(ann, src_len)

        if start >= end:
            print(f"  Skipping episode {ep_idx} for {task_name}: "
                  f"invalid range [{start}, {end})")
            continue

        # Load source parquet
        df = load_parquet(src_dir, ep_idx)
        sub_df = df.iloc[start:end].copy()
        n = len(sub_df)

        # Rebuild columns for the sub-episode
        new_ep_idx = ep_count
        timestamps = [(i / fps) for i in range(n)]
        frame_indices = list(range(n))
        global_indices = list(range(total_frames, total_frames + n))
        done_flags = [False] * (n - 1) + [True]

        # Get observation columns
        obs_states = sub_df["observation.state"].tolist()
        obs_eef_pos = sub_df["observation.eef_pos"].tolist()
        obs_eef_rot = sub_df["observation.eef_rot"].tolist()
        obs_eef_euler = sub_df["observation.eef_euler"].tolist()

        # Recompute actions with proper +1 shift
        abs_joints = []
        delta_joints = []
        abs_eefs = []
        delta_eefs = []

        for i in range(n):
            cur_s = list(obs_states[i])
            cur_eef_p = list(obs_eef_pos[i])
            cur_eef_e = list(obs_eef_euler[i])

            if i < n - 1:
                nxt_s = list(obs_states[i + 1])
                nxt_eef_p = list(obs_eef_pos[i + 1])
                nxt_eef_e = list(obs_eef_euler[i + 1])

                abs_joints.append(nxt_s)
                delta_joints.append([nxt_s[j] - cur_s[j] for j in range(7)])
                abs_eefs.append(list(nxt_eef_p) + list(nxt_eef_e) + [nxt_s[6]])
                delta_eefs.append(
                    [nxt_eef_p[j] - cur_eef_p[j] for j in range(3)]
                    + [nxt_eef_e[j] - cur_eef_e[j] for j in range(3)]
                    + [nxt_s[6] - cur_s[6]])
            else:
                abs_joints.append(cur_s)
                delta_joints.append([0.0] * 7)
                abs_eefs.append(list(cur_eef_p) + list(cur_eef_e) + [cur_s[6]])
                delta_eefs.append([0.0] * 7)

        # Build parquet
        import pyarrow as pa

        vp = (f"videos/chunk-000/observation.images.{cam_name}/"
              f"episode_{new_ep_idx:06d}.mp4")

        arrays = {
            "timestamp": pa.array(timestamps, type=pa.float32()),
            "episode_index": pa.array([new_ep_idx] * n, type=pa.int64()),
            "frame_index": pa.array(frame_indices, type=pa.int64()),
            "index": pa.array(global_indices, type=pa.int64()),
            "next.done": pa.array(done_flags, type=pa.bool_()),
            "task_index": pa.array([0] * n, type=pa.int64()),
            "observation.state": pa.array(
                [list(s) for s in obs_states], type=pa.list_(pa.float32())),
            "observation.eef_pos": pa.array(
                [list(p) for p in obs_eef_pos], type=pa.list_(pa.float32())),
            "observation.eef_rot": pa.array(
                [list(r) for r in obs_eef_rot], type=pa.list_(pa.float32())),
            "observation.eef_euler": pa.array(
                [list(e) for e in obs_eef_euler], type=pa.list_(pa.float32())),
            "action.abs_joint": pa.array(abs_joints, type=pa.list_(pa.float32())),
            "action.delta_joint": pa.array(delta_joints, type=pa.list_(pa.float32())),
            "action.abs_eef": pa.array(abs_eefs, type=pa.list_(pa.float32())),
            "action.delta_eef": pa.array(delta_eefs, type=pa.list_(pa.float32())),
            f"observation.images.{cam_name}": pa.StructArray.from_arrays(
                [pa.array([vp] * n, type=pa.string()),
                 pa.array(timestamps, type=pa.float32())],
                names=["path", "timestamp"]),
        }

        pq_path = os.path.join(out_dir, "data", "chunk-000",
                               f"episode_{new_ep_idx:06d}.parquet")
        pq.write_table(pa.table(arrays), pq_path)

        # Extract and write sub-video
        frames = load_video_frames(src_dir, ep_idx, cam_name)
        sub_frames = frames[start:end]
        if sub_frames:
            vout_path = os.path.join(out_dir, "videos", "chunk-000",
                                     f"observation.images.{cam_name}",
                                     f"episode_{new_ep_idx:06d}.mp4")
            h, w = sub_frames[0].shape[:2]
            writer = cv2.VideoWriter(vout_path,
                                     cv2.VideoWriter_fourcc(*'mp4v'),
                                     fps, (w, h))
            for fr in sub_frames:
                writer.write(fr)
            writer.release()

        print(f"  {task_name} episode {new_ep_idx}: "
              f"frames {start}-{end-1} from source ep {ep_idx} "
              f"({n} frames, {n/fps:.1f}s)")

        ep_count += 1
        total_frames += n

    # Write metadata
    _write_subtask_meta(out_dir, ep_count, total_frames, fps,
                        task_label, cam_name, src_info, annotations,
                        frame_selector)
    print(f"  {task_name}: {ep_count} episodes, {total_frames} frames total")


def _write_subtask_meta(out_dir, ep_count, total_frames, fps,
                        task_label, cam_name, src_info, annotations,
                        frame_selector):
    """Write info.json, episodes.jsonl, tasks.jsonl for a sub-task dataset."""
    meta_dir = os.path.join(out_dir, "meta")

    # Feature schema (same as source)
    features = {}
    for name, shape, nms in [
        ("observation.state", [7],
         ["j1", "j2", "j3", "j4", "j5", "j6", "gripper"]),
        ("observation.eef_pos", [3], ["x", "y", "z"]),
        ("observation.eef_rot", [9],
         ["r00", "r01", "r02", "r10", "r11", "r12", "r20", "r21", "r22"]),
        ("observation.eef_euler", [3], ["roll", "pitch", "yaw"]),
        ("action.abs_joint", [7],
         ["j1", "j2", "j3", "j4", "j5", "j6", "gripper"]),
        ("action.delta_joint", [7],
         ["dj1", "dj2", "dj3", "dj4", "dj5", "dj6", "dgripper"]),
        ("action.abs_eef", [7],
         ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]),
        ("action.delta_eef", [7],
         ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "dgripper"]),
    ]:
        features[name] = {"dtype": "float32", "shape": shape, "names": nms}

    features[f"observation.images.{cam_name}"] = {
        "dtype": "video", "shape": [480, 640, 3],
        "names": ["height", "width", "channel"],
        "info": {"video.fps": fps, "video.codec": "mp4v"},
    }
    for k, dt, sh in [
        ("episode_index", "int64", [1]), ("frame_index", "int64", [1]),
        ("timestamp", "float32", [1]), ("index", "int64", [1]),
        ("next.done", "bool", [1]), ("task_index", "int64", [1]),
    ]:
        features[k] = {"dtype": dt, "shape": sh}

    info = {
        "codebase_version": "v2.1",
        "robot_type": "piper_hanging",
        "fps": fps,
        "total_episodes": ep_count,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": ep_count,
        "splits": {"train": f"0:{ep_count}"},
        "features": features,
    }
    with open(os.path.join(meta_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=2)

    with open(os.path.join(meta_dir, "tasks.jsonl"), "w") as f:
        f.write(json.dumps({"task_index": 0, "task": task_label}) + "\n")

    # Per-episode metadata
    ep_i = 0
    with open(os.path.join(meta_dir, "episodes.jsonl"), "w") as f:
        for ann in annotations:
            start, end = frame_selector(ann, ann["length"])
            if start >= end:
                continue
            ep_len = end - start
            f.write(json.dumps({
                "episode_index": ep_i,
                "tasks": [task_label],
                "length": ep_len,
            }) + "\n")
            ep_i += 1


def main():
    parser = argparse.ArgumentParser(
        description="Partition full-task episodes into sub-task datasets")
    parser.add_argument("--src", required=True,
                        help="Source dataset directory (e.g. ./data/tool_good)")
    parser.add_argument("--annotations", default="splits.json",
                        help="JSON file to save/load split annotations")
    parser.add_argument("--export", default=None,
                        help="Export sub-datasets to this directory "
                             "(creates pick/, inspect/, place_good/ inside)")
    parser.add_argument("--cam", default="cam_0",
                        help="Camera name (default: cam_0)")
    parser.add_argument("--task-type", default="good",
                        choices=["good", "bad"],
                        help="Whether these demos are good or bad "
                             "(affects place sub-task naming)")
    args = parser.parse_args()

    src_dir = os.path.abspath(args.src)
    info, episodes = load_episode_info(src_dir)
    n_eps = len(episodes)
    print(f"Source dataset: {src_dir}")
    print(f"  {n_eps} episodes, {info['total_frames']} total frames, "
          f"{info['fps']} fps")

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
        # Export mode: use saved annotations to create sub-datasets
        if not saved:
            print("No annotations found! Run without --export first to "
                  "annotate episodes.")
            sys.exit(1)

        ann_list = sorted(saved.values(), key=lambda a: a["episode_index"])
        print(f"\nExporting {len(ann_list)} annotated episodes...")

        place_name = "place_good" if args.task_type == "good" else "place_bad"

        # PICK: frame 0 to pick_end (inclusive)
        pick_dir = os.path.join(args.export, "pick")
        print(f"\n--- PICK → {pick_dir} ---")
        export_subtask(
            src_dir, info, ann_list, "PICK", pick_dir,
            frame_selector=lambda a, _: (0, a["pick_end"] + 1),
            task_label="pick tool",
            cam_name=args.cam)

        # INSPECT: pick_end+1 to inspect_end (inclusive)
        inspect_dir = os.path.join(args.export, "inspect")
        print(f"\n--- INSPECT → {inspect_dir} ---")
        export_subtask(
            src_dir, info, ann_list, "INSPECT", inspect_dir,
            frame_selector=lambda a, _: (a["pick_end"] + 1,
                                         a["inspect_end"] + 1),
            task_label="inspect tool",
            cam_name=args.cam)

        # PLACE: inspect_end+1 to end
        place_dir = os.path.join(args.export, place_name)
        print(f"\n--- {place_name.upper()} → {place_dir} ---")
        export_subtask(
            src_dir, info, ann_list, place_name.upper(), place_dir,
            frame_selector=lambda a, l: (a["inspect_end"] + 1, l),
            task_label=f"place tool {args.task_type}",
            cam_name=args.cam)

        print("\nExport complete!")
        return

    # Annotation mode: go through each episode
    print(f"\nAnnotating episodes... ({len(saved)} already done)")
    print("Controls: Space=play, Arrows=jump 0.5s, 1=PICK end, "
          "2=INSPECT end, S=save, R=reset, Q=quit\n")

    for ep in episodes:
        ep_idx = ep["episode_index"]
        ep_len = ep["length"]

        if ep_idx in saved:
            s = saved[ep_idx]
            print(f"Episode {ep_idx} ({ep_len} frames): "
                  f"already annotated — PICK end={s['pick_end']}, "
                  f"INSPECT end={s['inspect_end']}")
            continue

        print(f"\nEpisode {ep_idx} ({ep_len} frames, "
              f"{ep_len / info['fps']:.1f}s):")
        frames = load_video_frames(src_dir, ep_idx, args.cam)

        result = annotate_episode(ep_idx, frames, ep_len)

        if result is None:
            # User quit
            print("\nQuitting. Saving annotations so far...")
            break
        elif result == "skip":
            continue
        else:
            saved[ep_idx] = {
                "episode_index": ep_idx,
                "length": ep_len,
                "pick_end": result["pick_end"],
                "inspect_end": result["inspect_end"],
            }
            print(f"  Saved: PICK [0-{result['pick_end']}], "
                  f"INSPECT [{result['pick_end']+1}-{result['inspect_end']}], "
                  f"PLACE [{result['inspect_end']+1}-{ep_len-1}]")

    # Save annotations (only if there's something to save)
    if saved:
        ann_list = sorted(saved.values(), key=lambda a: a["episode_index"])
        with open(ann_path, "w") as f:
            json.dump(ann_list, f, indent=2)
        print(f"\nSaved {len(ann_list)} annotations to {ann_path}")
    else:
        print("\nNo annotations to save.")


if __name__ == "__main__":
    main()
