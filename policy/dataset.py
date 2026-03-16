"""Dataset loader for our LeRobot v2.1 parquet + mp4 format."""

import json
import os

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset


class PiperDataset(Dataset):
    """Loads parquet + mp4 episodes and returns windowed observation-action pairs.

    Video frames are cached to disk as .npy files and memory-mapped,
    so RAM usage stays low regardless of dataset size.

    Each sample contains:
        obs_state:  (obs_horizon, 7) — past joint+gripper states
        obs_image:  (obs_horizon, 3, H, W) — past camera frames
        action:     (pred_horizon, 7) — future joint+gripper targets (abs_joint)
    """

    def __init__(self, dataset_dir, obs_horizon=2, pred_horizon=16,
                 image_size=(240, 320), cam_name="cam_0"):
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.image_size = image_size

        dataset_dir = os.path.abspath(dataset_dir)

        # Load metadata
        with open(os.path.join(dataset_dir, "meta", "info.json")) as f:
            info = json.load(f)
        self.fps = info["fps"]

        episodes = []
        with open(os.path.join(dataset_dir, "meta", "episodes.jsonl")) as f:
            for line in f:
                episodes.append(json.loads(line.strip()))

        # Frame cache directory (memory-mapped .npy files)
        cache_dir = os.path.join(dataset_dir, ".frame_cache",
                                 f"{image_size[0]}x{image_size[1]}")
        os.makedirs(cache_dir, exist_ok=True)

        # Load all episodes
        self.states = []     # list of (ep_len, 7) numpy arrays
        self.actions = []    # list of (ep_len, 7) numpy arrays
        self.frame_maps = [] # list of memory-mapped (ep_len, H, W, 3) uint8 arrays

        print(f"Loading {len(episodes)} episodes from {dataset_dir}...")

        for ep in episodes:
            ep_idx = ep["episode_index"]

            # Load parquet
            pq_path = os.path.join(dataset_dir, "data", "chunk-000",
                                   f"episode_{ep_idx:06d}.parquet")
            table = pq.read_table(pq_path)
            df = table.to_pandas()

            states = np.array(df["observation.state"].tolist(), dtype=np.float32)
            actions = np.array(df["action.abs_joint"].tolist(), dtype=np.float32)
            self.states.append(states)
            self.actions.append(actions)

            # Load or cache video frames as memory-mapped numpy
            cache_path = os.path.join(cache_dir, f"episode_{ep_idx:06d}.npy")
            vpath = os.path.join(dataset_dir, "videos", "chunk-000",
                                 f"observation.images.{cam_name}",
                                 f"episode_{ep_idx:06d}.mp4")

            if os.path.exists(cache_path):
                # Fast path: memory-map existing cache
                frames_mmap = np.load(cache_path, mmap_mode='r')
            else:
                # First run: extract frames, save to cache, then mmap
                frames = self._load_video(vpath, len(states))
                frames_arr = np.stack(frames)  # (T, H, W, 3) uint8
                np.save(cache_path, frames_arr)
                del frames, frames_arr
                frames_mmap = np.load(cache_path, mmap_mode='r')

            self.frame_maps.append(frames_mmap)

            if (ep_idx + 1) % 10 == 0:
                print(f"  Loaded {ep_idx + 1}/{len(episodes)} episodes")

        # Build sample index: (episode_idx, timestep) pairs
        self.samples = []
        for ep_idx in range(len(self.states)):
            ep_len = len(self.states[ep_idx])
            for t in range(obs_horizon - 1, ep_len - pred_horizon):
                self.samples.append((ep_idx, t))

        print(f"  Total: {len(self.samples)} training samples "
              f"from {len(episodes)} episodes")

    def _load_video(self, path, expected_frames):
        """Load and resize all video frames."""
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.image_size[1], self.image_size[0]),
                               interpolation=cv2.INTER_AREA)
            frames.append(frame)
        cap.release()

        # Handle frame count mismatch
        if len(frames) != expected_frames:
            n = min(len(frames), expected_frames)
            frames = frames[:n]
            if n < expected_frames:
                frames.extend([frames[-1]] * (expected_frames - n))

        return frames

    def get_all_states(self):
        """Return all states as a single (N, 7) numpy array for normalization."""
        return np.concatenate(self.states, axis=0)

    def get_all_actions(self):
        """Return all actions as a single (N, 7) numpy array for normalization."""
        return np.concatenate(self.actions, axis=0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ep_idx, t = self.samples[idx]
        To = self.obs_horizon
        Tp = self.pred_horizon

        # Observation: past To timesteps [t-To+1, ..., t]
        obs_state = self.states[ep_idx][t - To + 1: t + 1].copy()  # (To, 7)

        # Images: past To timesteps (from memory-mapped array)
        obs_images = np.empty((To, 3, *self.image_size), dtype=np.float32)
        for j, i in enumerate(range(t - To + 1, t + 1)):
            img = self.frame_maps[ep_idx][i]       # mmap read, uint8 (H, W, 3)
            obs_images[j] = np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.0

        # Action: next Tp timesteps [t+1, ..., t+Tp]
        action = self.actions[ep_idx][t: t + Tp].copy()  # (Tp, 7)

        return {
            "obs_state": torch.from_numpy(obs_state),
            "obs_image": torch.from_numpy(obs_images),
            "action": torch.from_numpy(action),
        }
