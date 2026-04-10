#!/usr/bin/env python3
"""
Zarr Episode Viewer — Browse and replay recorded demos visually + on robot.

Usage:
    # List all episodes
    python zarr_viewer.py --dataset ./data/demo_good --list

    # View-only (no robot)
    python zarr_viewer.py --dataset ./data/demo_good --episode 0

    # Replay on robot
    python zarr_viewer.py --dataset ./data/demo_good --episode 0 --can can0

    # Replay at half speed
    python zarr_viewer.py --dataset ./data/demo_good --episode 0 --can can0 --speed 0.5

    # Play all episodes on robot
    python zarr_viewer.py --dataset ./data/demo_good --all --can can0

Controls (during playback):
    SPACE  = pause / resume
    LEFT   = back 20 frames
    RIGHT  = forward 20 frames
    N      = next episode
    P      = previous episode
    Q/ESC  = quit
"""

import argparse
import math
import os
import sys
import time
import threading

import cv2
import numpy as np
import zarr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from piper_core import (
    PiperHangingController,
    HOME_POSITION,
    JOINT_LOWER,
    JOINT_UPPER,
    RAD_TO_MDEG,
    J6_PHYSICAL_OFFSET,
)

GRIPPER_MAX_UM = 70000  # 70mm in micrometers


# ── Replay controller (interpolated joint sender) ───────────────────────────

class ReplayController:
    """Sends joint commands to robot with linear interpolation."""

    def __init__(self, ctrl, hz=50):
        self._piper = ctrl.piper
        self._speed = ctrl.speed
        self._lock = threading.Lock()
        self._prev_q = None
        self._next_q = None
        self._prev_grip = 0.0
        self._next_grip = 0.0
        self._t_prev = 0.0
        self._t_next = 0.0
        self._running = True
        self._hz = hz
        threading.Thread(target=self._loop, daemon=True, name="replay-cmd").start()

    def set_target(self, q, grip, t):
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
                if pq is not None and tn > tp:
                    now = time.time()
                    alpha = min(1.0, max(0.0, (now - tp) / (tn - tp)))
                    q_interp = [pq[j] + alpha * (nq[j] - pq[j]) for j in range(len(nq))]
                    g_interp = pg + alpha * (ng - pg)
                else:
                    q_interp = nq
                    g_interp = ng
                try:
                    margin = math.radians(2.0)
                    q_clamped = [max(JOINT_LOWER[j] + margin,
                                     min(JOINT_UPPER[j] - margin, q_interp[j]))
                                 for j in range(6)]
                    q_physical = list(q_clamped)
                    q_physical[5] += J6_PHYSICAL_OFFSET
                    jcmds = [round(q * RAD_TO_MDEG) for q in q_physical]
                    self._piper.MotionCtrl_2(0x01, 0x01, self._speed, 0x00)
                    self._piper.JointCtrl(*jcmds)
                except Exception:
                    pass
                try:
                    self._piper.GripperCtrl(round(g_interp * GRIPPER_MAX_UM), 1000, 0x01, 0)
                except Exception:
                    pass
            time.sleep(1.0 / self._hz)


# ── Dataset loading ─────────────────────────────────────────────────────────

def load_dataset(dataset_dir):
    zarr_path = os.path.join(dataset_dir, "dataset.zarr")
    if not os.path.exists(zarr_path):
        print(f"  ERROR: {zarr_path} not found")
        sys.exit(1)
    root = zarr.open_group(zarr_path, mode="r")
    ends = root["meta"]["episode_ends"][:]
    starts = np.concatenate([[0], ends[:-1]])
    return root, starts, ends


def list_episodes(root, starts, ends):
    n = len(ends)
    print(f"\n  Dataset: {n} episodes, {int(ends[-1])} total frames\n")
    print(f"  {'Ep':>4s}  {'Frames':>7s}  {'Duration':>8s}")
    print(f"  {'─'*4}  {'─'*7}  {'─'*8}")

    meta_path = os.path.join(os.path.dirname(root.store.path), "meta", "info.json")
    fps = 20
    if os.path.exists(meta_path):
        import json
        with open(meta_path) as f:
            info = json.load(f)
            fps = info.get("fps", 20)

    for i in range(n):
        length = int(ends[i] - starts[i])
        dur = length / fps
        print(f"  {i:4d}  {length:7d}  {dur:7.1f}s")
    print()


# ── Playback ────────────────────────────────────────────────────────────────

def play_episode(root, starts, ends, ep_idx, speed, fps, replay_cmd=None):
    n_eps = len(ends)
    if ep_idx < 0 or ep_idx >= n_eps:
        print(f"  Episode {ep_idx} out of range (0-{n_eps-1})")
        return None

    s, e = int(starts[ep_idx]), int(ends[ep_idx])
    length = e - s
    data = root["data"]

    cam_keys = [k for k in data.keys() if k.startswith("img_")]
    cam_names = [k.replace("img_", "") for k in cam_keys]
    mode = "ROBOT" if replay_cmd else "VIEW"

    print(f"\n  Episode {ep_idx}: {length} frames ({length/fps:.1f}s) | "
          f"cameras: {cam_names} | speed: {speed}x | mode: {mode}")
    print(f"  Controls: SPACE=pause  ←→=seek  N/P=next/prev  Q=quit\n")

    interval = 1.0 / (fps * speed)
    paused = False
    frame_i = 0

    while True:
        idx = s + frame_i
        t_start = time.time()

        # Send joints to robot
        if replay_cmd and not paused:
            state = data["state"][idx]
            target_q = [float(v) for v in state[:6]]
            target_grip = float(state[6])
            replay_cmd.set_target(target_q, target_grip, time.time() + interval)

        # Load frames for all cameras
        panels = []
        for cam_key in cam_keys:
            img = data[cam_key][idx]  # (H, W, 3) uint8
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.shape[2] == 3 else img
            cam_name = cam_key.replace("img_", "")

            # Draw info overlay
            cv2.putText(img, f"ep:{ep_idx} f:{frame_i}/{length} {cam_name}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if paused:
                cv2.putText(img, "PAUSED", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Draw state info
            state = data["state"][idx]
            state_txt = "joints: " + " ".join(f"{v:.2f}" for v in state[:6])
            grip_txt = f"grip: {state[6]:.2f}"
            cv2.putText(img, state_txt, (10, img.shape[0] - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(img, grip_txt, (10, img.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Progress bar
            bar_y = img.shape[0] - 5
            bar_w = img.shape[1] - 20
            progress = frame_i / max(length - 1, 1)
            cv2.rectangle(img, (10, bar_y - 3), (10 + bar_w, bar_y + 3), (80, 80, 80), -1)
            cv2.rectangle(img, (10, bar_y - 3), (10 + int(bar_w * progress), bar_y + 3),
                          (0, 255, 0), -1)

            panels.append(img)

        # Stack side-by-side
        if len(panels) > 1:
            h = min(p.shape[0] for p in panels)
            resized = []
            for p in panels:
                if p.shape[0] != h:
                    w = int(p.shape[1] * h / p.shape[0])
                    p = cv2.resize(p, (w, h))
                resized.append(p)
            display = np.hstack(resized)
        elif panels:
            display = panels[0]
        else:
            break

        cv2.imshow("Zarr Episode Viewer", display)

        # Handle keys
        wait_ms = 1 if paused else max(1, int(interval * 1000))
        key = cv2.waitKey(wait_ms) & 0xFF

        if key == ord('q') or key == 27:  # Q or ESC
            cv2.destroyAllWindows()
            return "quit"
        elif key == ord(' '):
            paused = not paused
        elif key == ord('n'):
            return "next"
        elif key == ord('p'):
            return "prev"
        elif key == 81 or key == 2:  # LEFT arrow
            frame_i = max(0, frame_i - 20)
        elif key == 83 or key == 3:  # RIGHT arrow
            frame_i = min(length - 1, frame_i + 20)

        if not paused:
            frame_i += 1
            if frame_i >= length:
                print(f"  Episode {ep_idx} done.")
                return "next"

            # Rate limit
            elapsed = time.time() - t_start
            sleep = interval - elapsed
            if sleep > 0:
                time.sleep(sleep)

    return None


def main():
    p = argparse.ArgumentParser(description="Zarr episode viewer + robot replay")
    p.add_argument("--dataset", required=True, help="Path to dataset dir")
    p.add_argument("--episode", type=int, default=0, help="Episode index to play")
    p.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    p.add_argument("--fps", type=int, default=20, help="Recording FPS")
    p.add_argument("--list", action="store_true", help="List episodes and exit")
    p.add_argument("--all", action="store_true", help="Play all episodes sequentially")
    p.add_argument("--can", default=None, help="CAN port for robot replay (e.g. can0). Omit for view-only.")
    p.add_argument("--robot-speed", type=int, default=70, help="Robot speed %% (default: 70)")
    args = p.parse_args()

    root, starts, ends = load_dataset(args.dataset)

    if args.list:
        list_episodes(root, starts, ends)
        return

    # Robot setup
    ctrl = None
    replay_cmd = None
    if args.can:
        print(f"\n  Connecting to robot on {args.can} ...")
        ctrl = PiperHangingController(can_port=args.can)
        if not ctrl.connect():
            print("  Failed to connect.")
            sys.exit(1)
        ctrl.speed = args.robot_speed
        print("  Connected. Homing ...")
        ctrl.go_home()
        print("  Ready.")
        replay_cmd = ReplayController(ctrl)

    n_eps = len(ends)
    ep = args.episode

    try:
        if args.all:
            for i in range(n_eps):
                result = play_episode(root, starts, ends, i, args.speed, args.fps, replay_cmd)
                if result == "quit":
                    break
        else:
            while True:
                result = play_episode(root, starts, ends, ep, args.speed, args.fps, replay_cmd)
                if result == "quit" or result is None:
                    break
                elif result == "next":
                    ep = min(ep + 1, n_eps - 1)
                elif result == "prev":
                    ep = max(ep - 1, 0)
    finally:
        if replay_cmd:
            replay_cmd.stop()
        if ctrl:
            print("\n  Homing robot ...")
            ctrl.go_home()
            print("  Relaxing ...")
            ctrl.go_relax()
            ctrl.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
