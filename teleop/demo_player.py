#!/usr/bin/env python3
"""
Demo Player — Replay recorded teleop episodes on the robot
=============================================================
Reads a LeRobot v2.1 dataset and replays an episode by sending
recorded joint targets to the robot at the original FPS.

Optionally displays camera footage from the recording alongside
live camera feeds (if cameras are connected).

Usage:
    # Replay episode 0 on the robot
    python3 demo_player.py --can can0 --dataset ./data/my_task --episode 0

    # Replay using delta joint actions instead of absolute
    python3 demo_player.py --can can0 --dataset ./data/my_task --episode 0 --action delta_joint

    # Replay with live camera + recorded camera comparison
    python3 demo_player.py --can can0 --dataset ./data/my_task --episode 0 --cameras 0

    # Dry run (no robot, just show what would happen)
    python3 demo_player.py --dataset ./data/my_task --episode 0 --dry-run

    # List all episodes in a dataset
    python3 demo_player.py --dataset ./data/my_task --list

Action modes:
    abs_joint    — Send absolute joint positions directly (default, most accurate)
    delta_joint  — Accumulate delta joint actions from home position
    abs_eef      — Send absolute EEF pose via firmware IK (Cartesian control)
"""

import argparse
import json
import math
import os
import sys
import time
import threading

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from piper_core import (
    PiperHangingController,
    HOME_POSITION,
    RAD_TO_MDEG,
    forward_kinematics,
    rotation_to_euler,
)

try:
    import cv2
    _cv2_ok = True
except ImportError:
    _cv2_ok = False

try:
    import pyarrow.parquet as pq
    _arrow_ok = True
except ImportError:
    _arrow_ok = False
    print("ERROR: pyarrow is required. Install with: pip install pyarrow")
    sys.exit(1)


# ── ANSI colors ──────────────────────────────────────────────────────────────
GREEN  = "\033[32m"
RED    = "\033[31m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


# ── Dataset reader ───────────────────────────────────────────────────────────

class DatasetReader:
    """Read a LeRobot v2.1 dataset."""

    def __init__(self, dataset_dir):
        self.dataset_dir = os.path.abspath(dataset_dir)
        self.meta_dir = os.path.join(self.dataset_dir, "meta")

        # Load info
        info_path = os.path.join(self.meta_dir, "info.json")
        if not os.path.exists(info_path):
            raise FileNotFoundError(f"No info.json found in {self.meta_dir}")
        with open(info_path) as f:
            self.info = json.load(f)

        self.fps = self.info.get("fps", 20)
        self.total_episodes = self.info.get("total_episodes", 0)
        self.total_frames = self.info.get("total_frames", 0)

        # Load episodes metadata
        self.episodes = []
        episodes_path = os.path.join(self.meta_dir, "episodes.jsonl")
        if os.path.exists(episodes_path):
            with open(episodes_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.episodes.append(json.loads(line))

        # Load tasks
        self.tasks = []
        tasks_path = os.path.join(self.meta_dir, "tasks.jsonl")
        if os.path.exists(tasks_path):
            with open(tasks_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.tasks.append(json.loads(line))

    def load_episode(self, episode_idx):
        """Load all rows for an episode. Returns list of dicts."""
        parquet_path = os.path.join(
            self.dataset_dir, "data", "chunk-000",
            f"episode_{episode_idx:06d}.parquet")
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Episode file not found: {parquet_path}")
        table = pq.read_table(parquet_path)
        return table.to_pydict()

    def get_video_path(self, episode_idx, camera_name):
        """Get path to video file for an episode + camera."""
        return os.path.join(
            self.dataset_dir, "videos", "chunk-000",
            f"observation.images.{camera_name}",
            f"episode_{episode_idx:06d}.mp4")

    def list_cameras(self):
        """List camera names from the dataset features."""
        cams = []
        for key in self.info.get("features", {}):
            if key.startswith("observation.images."):
                cams.append(key.replace("observation.images.", ""))
        return cams


# ── Replay controller ────────────────────────────────────────────────────────

class ReplayController:
    """Sends joint commands to robot with linear interpolation for smoothness."""

    def __init__(self, ctrl, hz=50):
        self._piper = ctrl.piper
        self._speed = ctrl.speed
        self._lock = threading.Lock()
        self._prev_q = None       # previous keyframe joints
        self._next_q = None       # next keyframe joints
        self._prev_grip = 0.0
        self._next_grip = 0.0
        self._t_prev = 0.0        # time of previous keyframe
        self._t_next = 0.0        # time of next keyframe
        self._running = True
        self._hz = hz
        threading.Thread(target=self._loop, daemon=True, name="replay-cmd").start()

    def set_target(self, q, grip, t):
        """Set next keyframe. Previous next becomes the new prev."""
        with self._lock:
            self._prev_q = self._next_q
            self._prev_grip = self._next_grip
            self._t_prev = self._t_next
            self._next_q = [float(v) for v in q]
            self._next_grip = float(grip)
            self._t_next = t

    def stop(self):
        self._running = False

    def _loop(self):
        while self._running:
            with self._lock:
                pq = self._prev_q
                nq = self._next_q
                pg = self._prev_grip
                ng = self._next_grip
                tp = self._t_prev
                tn = self._t_next

            if nq is not None:
                # Interpolate between prev and next keyframe
                if pq is not None and tn > tp:
                    now = time.time()
                    alpha = min(1.0, max(0.0, (now - tp) / (tn - tp)))
                    q_interp = [pq[j] + alpha * (nq[j] - pq[j]) for j in range(len(nq))]
                    g_interp = pg + alpha * (ng - pg)
                else:
                    q_interp = nq
                    g_interp = ng
                try:
                    jcmds = [round(q * RAD_TO_MDEG) for q in q_interp]
                    self._piper.MotionCtrl_2(0x01, 0x01, self._speed, 0x00)
                    self._piper.JointCtrl(*jcmds)
                    self._piper.GripperCtrl(round(g_interp * 35000), 1000, 0x01, 0)
                except Exception:
                    pass
            time.sleep(1.0 / self._hz)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Replay recorded teleop demos on the robot.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", required=True, help="Dataset directory")
    p.add_argument("--episode", type=int, default=0, help="Episode index to replay")
    p.add_argument("--can", default=None, help="CAN interface (e.g. can0)")
    p.add_argument("--action", default="abs_joint",
                   choices=["abs_joint", "delta_joint", "abs_eef"],
                   help="Action representation to use for replay")
    p.add_argument("--speed", type=int, default=50, help="Robot speed %%")
    p.add_argument("--cameras", default="",
                   help="Live camera IDs for side-by-side comparison (e.g. 0)")
    p.add_argument("--dry-run", action="store_true",
                   help="Don't send commands to robot, just print")
    p.add_argument("--list", action="store_true",
                   help="List all episodes and exit")
    p.add_argument("--slow", type=float, default=1.0,
                   help="Slowdown factor (2.0 = half speed)")
    args = p.parse_args()

    # Load dataset
    reader = DatasetReader(args.dataset)
    print()
    print("=" * 62)
    print(f"  {BOLD}DEMO PLAYER{RESET}")
    print("=" * 62)
    print(f"  Dataset  : {args.dataset}")
    print(f"  Episodes : {reader.total_episodes}")
    print(f"  Frames   : {reader.total_frames}")
    print(f"  FPS      : {reader.fps}")
    if reader.tasks:
        print(f"  Task     : {reader.tasks[0].get('task', 'N/A')}")
    print()

    # List mode
    if args.list:
        print(f"  {'Ep':>4s}  {'Frames':>7s}  {'Duration':>8s}  Task")
        print(f"  {'─'*4}  {'─'*7}  {'─'*8}  {'─'*30}")
        for ep in reader.episodes:
            idx = ep["episode_index"]
            length = ep["length"]
            dur = length / reader.fps
            task = ep.get("tasks", [""])[0] if ep.get("tasks") else ""
            print(f"  {idx:4d}  {length:7d}  {dur:7.1f}s  {task}")
        print()
        return

    # Load episode
    ep_idx = args.episode
    if ep_idx >= reader.total_episodes:
        print(f"  {RED}ERROR: Episode {ep_idx} not found "
              f"(dataset has {reader.total_episodes} episodes){RESET}")
        sys.exit(1)

    print(f"  Loading episode {ep_idx} ...")
    data = reader.load_episode(ep_idx)
    n_frames = len(data["timestamp"])
    duration = data["timestamp"][-1]
    print(f"  {n_frames} frames, {duration:.1f}s")

    # Get action data based on mode
    action_key = f"action.{args.action}"
    if action_key not in data:
        print(f"  {RED}ERROR: Action '{action_key}' not found in dataset.{RESET}")
        print(f"  Available: {[k for k in data.keys() if k.startswith('action.')]}")
        sys.exit(1)
    actions = data[action_key]
    states = data.get("observation.state", [])

    print(f"  Action   : {action_key}")
    print(f"  Mode     : {'DRY RUN' if args.dry_run else 'ROBOT'}")
    if args.slow != 1.0:
        print(f"  Slowdown : {args.slow}x")
    print()

    # Open recorded videos
    rec_videos = {}
    rec_caps = {}
    dataset_cams = reader.list_cameras()
    for cam in dataset_cams:
        vpath = reader.get_video_path(ep_idx, cam)
        if os.path.exists(vpath) and _cv2_ok:
            cap = cv2.VideoCapture(vpath)
            if cap.isOpened():
                rec_caps[cam] = cap
                print(f"  Recorded video: {cam} ({vpath})")

    # Open live cameras
    live_caps = {}
    live_cam_ids = [int(x) for x in args.cameras.split(",") if x.strip()]
    if _cv2_ok:
        for cid in live_cam_ids:
            cap = cv2.VideoCapture(cid)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                live_caps[f"live_{cid}"] = cap
                print(f"  Live camera: {cid}")

    # Connect robot
    ctrl = None
    replay_cmd = None
    if not args.dry_run:
        if args.can is None:
            print(f"  {RED}ERROR: --can required for robot replay (or use --dry-run){RESET}")
            sys.exit(1)
        print(f"\n  Connecting to robot on {args.can} ...")
        ctrl = PiperHangingController(can_port=args.can)
        if not ctrl.connect():
            print(f"  {RED}Failed to connect.{RESET}")
            sys.exit(1)
        ctrl.speed = args.speed
        print(f"  {GREEN}Connected.{RESET}")

        # Home first
        print("  Homing robot ...")
        ctrl.go_home()
        print("  Ready.")
        replay_cmd = ReplayController(ctrl)

    # ── Replay loop ──────────────────────────────────────────────────────
    print(f"\n  {GREEN}▶ Playing episode {ep_idx}{RESET} "
          f"({n_frames} frames, {duration:.1f}s)")
    print(f"  Press Ctrl+C to stop\n")

    interval = (1.0 / reader.fps) * args.slow

    # For delta mode: start from the first frame's actual joint state, not HOME
    if args.action == "delta_joint" and states:
        first_state = [float(v) for v in states[0]]
        current_q = first_state[:6] + [first_state[6] if len(first_state) > 6 else 0.0]
    else:
        current_q = list(HOME_POSITION) + [0.0]

    try:
        for i in range(n_frames):
            t_start = time.time()

            action = actions[i]
            if isinstance(action, dict):
                action = list(action.values())
            action = [float(v) for v in action]

            # Compute joint target based on action mode
            if args.action == "abs_joint":
                target_q = action[:6]
                target_grip = action[6] if len(action) > 6 else 0.0

            elif args.action == "delta_joint":
                for j in range(6):
                    current_q[j] += action[j]
                if len(action) > 6:
                    current_q[6] += action[6]
                target_q = current_q[:6]
                target_grip = current_q[6]

            elif args.action == "abs_eef":
                # For EEF mode, we'd need to use IK or firmware Cartesian
                # For now, fall back to reading the state column
                if states:
                    st = states[i]
                    target_q = [float(v) for v in st[:6]]
                    target_grip = float(st[6]) if len(st) > 6 else 0.0
                else:
                    print(f"  {RED}abs_eef requires observation.state in dataset{RESET}")
                    break

            # Send to robot (with interpolation timestamp)
            if replay_cmd is not None:
                replay_cmd.set_target(target_q, target_grip, time.time() + interval)

            # Display
            timestamp = float(data["timestamp"][i])
            if i % max(1, reader.fps // 5) == 0 or i == n_frames - 1:
                q_str = " ".join(f"{math.degrees(q):+6.1f}" for q in target_q)
                progress = (i + 1) / n_frames * 100
                print(f"\r  [{progress:5.1f}%] t={timestamp:6.2f}s  "
                      f"joints: {q_str}  grip: {target_grip:.0%}   ",
                      end="", flush=True)

            # Show video frames
            if _cv2_ok and (rec_caps or live_caps):
                panels = []
                # Recorded video frame
                for cam, cap in rec_caps.items():
                    ret, frame = cap.read()
                    if ret:
                        cv2.putText(frame, f"REC {cam} t={timestamp:.2f}s",
                                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (0, 0, 255), 2)
                        panels.append(frame)

                # Live camera frame
                for cam, cap in live_caps.items():
                    ret, frame = cap.read()
                    if ret:
                        cv2.putText(frame, f"LIVE {cam}",
                                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (0, 255, 0), 2)
                        panels.append(frame)

                if panels:
                    # Resize all to same height
                    h = min(p.shape[0] for p in panels)
                    resized = []
                    for panel in panels:
                        scale = h / panel.shape[0]
                        w = int(panel.shape[1] * scale)
                        resized.append(cv2.resize(panel, (w, h)))
                    combined = np.hstack(resized)
                    cv2.imshow("Demo Player", combined)
                    if cv2.waitKey(1) == 27:  # ESC
                        print("\n\n  Stopped by ESC.")
                        break

            # Rate limit
            elapsed = time.time() - t_start
            sleep_t = interval - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

        print(f"\n\n  {GREEN}✓ Replay complete.{RESET}")

    except KeyboardInterrupt:
        print(f"\n\n  Stopped at frame {i}/{n_frames}.")

    finally:
        if replay_cmd:
            replay_cmd.stop()

        # Go home after replay
        if ctrl:
            print("  Homing robot ...")
            ctrl.go_home()
            print("  Relaxing ...")
            ctrl.go_relax()
            ctrl.shutdown()

        for cap in rec_caps.values():
            cap.release()
        for cap in live_caps.values():
            cap.release()
        if _cv2_ok:
            cv2.destroyAllWindows()

    print("  Done.")


if __name__ == "__main__":
    main()
