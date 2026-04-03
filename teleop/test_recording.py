#!/usr/bin/env python3
"""Test the Zarr recording pipeline with real cameras + robot (no movement).

Connects to the robot arm, reads its current joint state, captures frames
from both cameras, records 3 short episodes, then verifies the Zarr store.

Usage:
  python teleop/test_recording.py --can can0
  python teleop/test_recording.py --can can0 --global-cam 0 --wrist-cam 2
  python teleop/test_recording.py --can can0 --episodes 5 --seconds 3
"""

import argparse
import json
import math
import os
import shutil
import sys
import time

import cv2
import numpy as np
import zarr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from piper_core import forward_kinematics, rotation_to_euler, PiperHangingController

# Reuse the recording classes from teleop
from numcodecs import Blosc

_zarr_ok = True


# ── Copied recording classes (self-contained) ───────────────────────────────

class EpisodeBuffer:
    def __init__(self):
        self.rows = []
        self.video_frames = {}
        self.t0 = None

    def add(self, joints, eef_pos, eef_rot_flat, eef_euler, gripper, camera_frames):
        now = time.time()
        if self.t0 is None:
            self.t0 = now
        self.rows.append({
            "timestamp": now - self.t0,
            "obs_state": list(joints) + [gripper],
            "obs_eef_pos": list(eef_pos),
            "obs_eef_rot": list(eef_rot_flat),
            "obs_eef_euler": list(eef_euler),
        })
        for name, fr in camera_frames.items():
            if name not in self.video_frames:
                self.video_frames[name] = []
            self.video_frames[name].append(fr)

    def __len__(self):
        return len(self.rows)


class CameraCapture:
    """Background-threaded camera capture for low-latency dual-camera reads."""

    def __init__(self, cam_map):
        import threading as _thr
        self._threading = _thr
        self.caps = {}
        self.names = []
        self._frames = {}
        self._locks = {}
        self._threads = []
        self._running = True
        for name, cid in cam_map.items():
            cap = cv2.VideoCapture(cid)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                self.caps[name] = cap
                self.names.append(name)
                self._frames[name] = None
                self._locks[name] = _thr.Lock()
                t = _thr.Thread(target=self._grab_loop, args=(name, cap),
                                daemon=True)
                t.start()
                self._threads.append(t)
                print(f"  {name}: dev {cid} opened ({int(actual_w)}x{int(actual_h)})")
            else:
                print(f"  WARNING: {name} camera (dev {cid}) failed to open")

    def _grab_loop(self, name, cap):
        while self._running:
            ret, frame = cap.read()
            if ret:
                with self._locks[name]:
                    self._frames[name] = frame

    def read_all(self):
        out = {}
        for name in self.names:
            with self._locks[name]:
                f = self._frames[name]
            if f is not None:
                out[name] = f
        return out

    def release(self):
        self._running = False
        for t in self._threads:
            t.join(timeout=1.0)
        for cap in self.caps.values():
            cap.release()


class ZarrDatasetWriter:
    IMG_HEIGHT = 720
    IMG_WIDTH = 1280

    def __init__(self, dataset_dir, fps, task, camera_names):
        self.dir = os.path.abspath(dataset_dir)
        self.fps = fps
        self.task = task
        self.cam_names = list(camera_names)

        zarr_path = os.path.join(self.dir, "dataset.zarr")
        self.root = zarr.open_group(zarr_path, mode="a")

        compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
        img_compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE)

        if "data" not in self.root:
            self.root.create_group("data")
        if "meta" not in self.root:
            self.root.create_group("meta")
        data = self.root["data"]
        meta = self.root["meta"]

        def _ensure(grp, name, shape, chunks, dtype, comp):
            if name not in grp:
                grp.create_dataset(name, shape=shape, chunks=chunks,
                                   dtype=dtype, compressor=comp)

        _ensure(data, "state",     (0, 7), (256, 7), "float32", compressor)
        _ensure(data, "action",    (0, 7), (256, 7), "float32", compressor)
        _ensure(data, "eef_pos",   (0, 3), (256, 3), "float32", compressor)
        _ensure(data, "eef_euler", (0, 3), (256, 3), "float32", compressor)
        _ensure(data, "timestamp", (0,),   (256,),   "float32", compressor)
        for cam in self.cam_names:
            _ensure(data, f"img_{cam}",
                    (0, self.IMG_HEIGHT, self.IMG_WIDTH, 3),
                    (1, self.IMG_HEIGHT, self.IMG_WIDTH, 3),
                    "uint8", img_compressor)
        _ensure(meta, "episode_ends", (0,), (256,), "int64", compressor)

        self.ep_count = len(meta["episode_ends"])
        self.total_frames = int(meta["episode_ends"][-1]) if self.ep_count > 0 else 0

    def save(self, buf):
        n = len(buf)
        if n == 0:
            return None
        ep = self.ep_count
        data = self.root["data"]
        meta = self.root["meta"]

        states = np.array([r["obs_state"] for r in buf.rows], dtype=np.float32)
        eef_pos = np.array([r["obs_eef_pos"] for r in buf.rows], dtype=np.float32)
        eef_euler = np.array([r["obs_eef_euler"] for r in buf.rows], dtype=np.float32)
        timestamps = np.array([r["timestamp"] for r in buf.rows], dtype=np.float32)

        actions = np.empty((n, 7), dtype=np.float32)
        for i in range(n - 1):
            actions[i] = buf.rows[i + 1]["obs_state"]
        actions[n - 1] = buf.rows[n - 1]["obs_state"]

        data["state"].append(states)
        data["action"].append(actions)
        data["eef_pos"].append(eef_pos)
        data["eef_euler"].append(eef_euler)
        data["timestamp"].append(timestamps)

        for cam in self.cam_names:
            frames = buf.video_frames.get(cam, [])
            if len(frames) >= n:
                img_arr = np.stack(frames[:n])
            elif frames:
                padded = frames + [frames[-1]] * (n - len(frames))
                img_arr = np.stack(padded)
            else:
                img_arr = np.zeros((n, self.IMG_HEIGHT, self.IMG_WIDTH, 3),
                                   dtype=np.uint8)
            data[f"img_{cam}"].append(img_arr)

        new_end = self.total_frames + n
        meta["episode_ends"].append(np.array([new_end], dtype=np.int64))
        self.ep_count += 1
        self.total_frames = new_end
        self._write_meta()
        return ep

    def _write_meta(self):
        info = {
            "codebase_version": "zarr_v1",
            "robot_type": "piper_hanging",
            "fps": self.fps,
            "task": self.task,
            "total_episodes": self.ep_count,
            "total_frames": self.total_frames,
            "cameras": self.cam_names,
            "image_shape": [self.IMG_HEIGHT, self.IMG_WIDTH, 3],
            "state_dim": 7,
            "action_dim": 7,
        }
        os.makedirs(os.path.join(self.dir, "meta"), exist_ok=True)
        with open(os.path.join(self.dir, "meta", "info.json"), "w") as f:
            json.dump(info, f, indent=2)


# ── Test logic ───────────────────────────────────────────────────────────────

def detect_cameras():
    """Scan for available cameras, return list of device IDs."""
    found = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            found.append(i)
            cap.release()
        else:
            cap.release()
    return found


def record_one_episode(ctrl, cameras, rec_fps, duration_s):
    """Record a single episode: read joints + cameras at rec_fps for duration_s."""
    buf = EpisodeBuffer()
    interval = 1.0 / rec_fps
    n_frames = int(duration_s * rec_fps)

    for i in range(n_frames):
        t0 = time.time()

        # Read robot state
        joints = ctrl.read_joints()
        T_ee, _ = forward_kinematics(joints)
        eef_p = T_ee[:3, 3].tolist()
        eef_R = T_ee[:3, :3]
        eef_rot_flat = eef_R.flatten().tolist()
        eef_eul = list(rotation_to_euler(eef_R))
        gripper = 0.0  # static — not moving

        # Read cameras
        cam_frames = cameras.read_all()

        buf.add(joints, eef_p, eef_rot_flat, eef_eul, gripper, cam_frames)

        # Show live feed
        for cname, frame in cam_frames.items():
            display = frame.copy()
            cv2.putText(display, f"REC {cname} [{i+1}/{n_frames}]",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 255), 2)
            cv2.imshow(cname, display)
        cv2.waitKey(1)

        elapsed = time.time() - t0
        if elapsed < interval:
            time.sleep(interval - elapsed)

    return buf


def verify_zarr(dataset_dir, expected_eps, cam_names):
    """Read back the Zarr store and print diagnostics."""
    zarr_path = os.path.join(dataset_dir, "dataset.zarr")
    root = zarr.open_group(zarr_path, mode="r")
    data = root["data"]
    meta = root["meta"]

    episode_ends = meta["episode_ends"][:]
    total = int(episode_ends[-1]) if len(episode_ends) > 0 else 0

    print(f"\n{'='*60}")
    print(f"  VERIFICATION — {zarr_path}")
    print(f"{'='*60}")
    print(f"  Episodes:     {len(episode_ends)} (expected {expected_eps})")
    print(f"  Total frames: {total}")
    print(f"  episode_ends: {episode_ends.tolist()}")
    print()

    for key in ["state", "action", "eef_pos", "eef_euler", "timestamp"]:
        arr = data[key]
        print(f"  data/{key:12s}  shape={str(arr.shape):16s}  dtype={arr.dtype}")

    for cam in cam_names:
        key = f"img_{cam}"
        if key in data:
            arr = data[key]
            print(f"  data/{key:12s}  shape={str(arr.shape):16s}  dtype={arr.dtype}")
            # Check a frame is not all zeros
            sample = arr[0]
            nonzero = np.count_nonzero(sample)
            total_px = sample.size
            print(f"    frame[0] nonzero: {nonzero}/{total_px} "
                  f"({100*nonzero/total_px:.1f}%)")

    # Verify state values are reasonable
    states = data["state"][:]
    print(f"\n  State stats (first episode):")
    ep0_end = int(episode_ends[0])
    s0 = states[:ep0_end]
    for j in range(7):
        name = f"j{j+1}" if j < 6 else "grip"
        print(f"    {name}: min={s0[:, j].min():.4f}  "
              f"max={s0[:, j].max():.4f}  "
              f"std={s0[:, j].std():.6f}")

    # Check actions have proper +1 shift
    actions = data["action"][:]
    a0 = actions[:ep0_end]
    s_shifted = states[1:ep0_end]
    match = np.allclose(a0[:ep0_end-1], s_shifted, atol=1e-6)
    print(f"\n  Action shift check (action[t] == state[t+1]): "
          f"{'PASS' if match else 'FAIL'}")

    # Disk size
    zarr_size = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, filenames in os.walk(zarr_path)
        for f in filenames
    )
    print(f"\n  Zarr store size: {zarr_size / 1024 / 1024:.1f} MB")

    # Load meta JSON
    info_path = os.path.join(dataset_dir, "meta", "info.json")
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
        print(f"  meta/info.json: {json.dumps(info, indent=4)}")

    ok = (len(episode_ends) == expected_eps and total > 0 and match)
    print(f"\n  {'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    print(f"{'='*60}\n")
    return ok


def main():
    parser = argparse.ArgumentParser(
        description="Test Zarr recording pipeline (robot stationary)")
    parser.add_argument("--can", default="can0", help="CAN port for robot arm")
    parser.add_argument("--global-cam", default="auto",
                        help="Global camera device ID (default: auto)")
    parser.add_argument("--wrist-cam", default="auto",
                        help="Wrist camera device ID (default: auto)")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of test episodes to record (default: 3)")
    parser.add_argument("--seconds", type=float, default=2.0,
                        help="Duration of each episode in seconds (default: 2)")
    parser.add_argument("--fps", type=int, default=20,
                        help="Recording FPS (default: 20)")
    parser.add_argument("--output", default="./data/_test_recording",
                        help="Output directory (default: ./data/_test_recording)")
    parser.add_argument("--keep", action="store_true",
                        help="Keep test data after verification (default: delete)")
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  ZARR RECORDING PIPELINE TEST")
    print("=" * 60)

    # ── Detect cameras ────────────────────────────────────────────────────
    print("\n  Scanning cameras...")
    available = detect_cameras()
    print(f"  Available devices: {available}")

    cam_map = {}
    if args.global_cam != "auto":
        cam_map["global"] = int(args.global_cam)
    elif available:
        cam_map["global"] = available[0]

    if args.wrist_cam != "auto":
        cam_map["wrist"] = int(args.wrist_cam)
    elif len(available) > 1:
        for cid in available:
            if cid != cam_map.get("global"):
                cam_map["wrist"] = cid
                break

    if not cam_map:
        print("  ERROR: No cameras found!")
        sys.exit(1)

    print(f"\n  Camera assignment: {cam_map}")
    cameras = CameraCapture(cam_map)
    if not cameras.names:
        print("  ERROR: No cameras could be opened!")
        sys.exit(1)

    # ── Warm up cameras (first few frames are often dark/corrupt) ─────────
    print("  Warming up cameras...")
    for _ in range(10):
        cameras.read_all()
        time.sleep(0.05)

    # ── Connect robot ─────────────────────────────────────────────────────
    print(f"\n  Connecting to robot on {args.can}...")
    ctrl = PiperHangingController(can_port=args.can)
    if not ctrl.connect():
        print("  ERROR: Failed to connect to robot!")
        cameras.release()
        sys.exit(1)
    ctrl.speed = 50
    print("  Robot connected.")

    joints = ctrl.read_joints()
    print(f"  Current joints: [{', '.join(f'{q:.4f}' for q in joints)}]")

    T_ee, _ = forward_kinematics(joints)
    pos = T_ee[:3, 3]
    print(f"  EEF position: ({pos[0]*100:.2f}, {pos[1]*100:.2f}, {pos[2]*100:.2f}) cm")

    # ── Clean output directory ────────────────────────────────────────────
    out_dir = os.path.abspath(args.output)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    print(f"\n  Output: {out_dir}")

    # ── Create writer ─────────────────────────────────────────────────────
    writer = ZarrDatasetWriter(out_dir, args.fps, "recording test", cameras.names)
    print(f"  Writer ready: fps={args.fps}, cameras={cameras.names}")

    # ── Record episodes ───────────────────────────────────────────────────
    print(f"\n  Recording {args.episodes} episodes x {args.seconds}s @ {args.fps} Hz")
    print(f"  (robot stationary — just reading state + cameras)\n")

    for ep in range(args.episodes):
        print(f"  --- Episode {ep} ---")
        t0 = time.time()
        buf = record_one_episode(ctrl, cameras, args.fps, args.seconds)
        rec_time = time.time() - t0
        print(f"    Captured {len(buf)} frames in {rec_time:.2f}s")

        t0 = time.time()
        ep_idx = writer.save(buf)
        save_time = time.time() - t0
        print(f"    Saved episode {ep_idx} in {save_time:.2f}s "
              f"({writer.total_frames} total frames)")

        # Brief pause between episodes
        if ep < args.episodes - 1:
            time.sleep(0.5)

    # ── Cleanup hardware ──────────────────────────────────────────────────
    cameras.release()
    cv2.destroyAllWindows()
    ctrl.shutdown()

    # ── Verify ────────────────────────────────────────────────────────────
    ok = verify_zarr(out_dir, args.episodes, cameras.names)

    # ── Cleanup test data ─────────────────────────────────────────────────
    if not args.keep:
        shutil.rmtree(out_dir)
        print(f"  Cleaned up {out_dir}")
    else:
        print(f"  Test data kept at {out_dir}")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
