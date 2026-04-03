"""Dataset loader for Zarr diffusion policy format with dual cameras.

Zarr structure:
    dataset.zarr/
    ├── data/
    │   ├── state       (N, 7)          float32
    │   ├── action      (N, 7)          float32
    │   ├── img_global  (N, 720, 1280, 3) uint8
    │   ├── img_wrist   (N, 720, 1280, 3) uint8
    │   ├── eef_pos     (N, 3)          float32
    │   ├── eef_euler   (N, 3)          float32
    │   └── timestamp   (N,)            float32
    └── meta/
        └── episode_ends (E,)           int64
"""

import json
import os

import cv2
import numpy as np
import torch
import zarr
from torch.utils.data import Dataset


class PiperDataset(Dataset):
    """Loads Zarr episodes and returns windowed observation-action pairs.

    Images are loaded on-the-fly from Zarr (memory-mapped + compressed),
    keeping RAM usage low regardless of dataset size.

    Each sample contains:
        obs_state:       (obs_horizon, 7)       — past joint+gripper states
        obs_img_global:  (obs_horizon, 3, H, W) — past global camera frames
        obs_img_wrist:   (obs_horizon, 3, H, W) — past wrist camera frames
        action:          (pred_horizon, 7)       — future joint+gripper targets
    """

    def __init__(self, dataset_dir, obs_horizon=2, pred_horizon=16,
                 image_size=(240, 426), camera_names=("global", "wrist")):
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.image_size = image_size
        self.camera_names = list(camera_names)

        dataset_dir = os.path.abspath(dataset_dir)
        zarr_path = os.path.join(dataset_dir, "dataset.zarr")
        self.root = zarr.open_group(zarr_path, mode="r")
        data = self.root["data"]
        meta = self.root["meta"]

        # Load metadata
        info_path = os.path.join(dataset_dir, "meta", "info.json")
        if os.path.exists(info_path):
            with open(info_path) as f:
                info = json.load(f)
            self.fps = info.get("fps", 20)
        else:
            self.fps = 20

        # Episode boundaries
        self.episode_ends = np.array(meta["episode_ends"][:], dtype=np.int64)
        self.num_episodes = len(self.episode_ends)
        self.total_frames = int(self.episode_ends[-1]) if self.num_episodes > 0 else 0

        # Zarr arrays (lazy — accessed on demand)
        self.states = data["state"]
        self.actions = data["action"]
        self.img_arrays = {}
        for cam in self.camera_names:
            key = f"img_{cam}"
            if key in data:
                self.img_arrays[cam] = data[key]

        # Filter camera_names to only those present in data
        self.camera_names = [c for c in self.camera_names if c in self.img_arrays]

        # Build sample index: (episode_start, episode_end, timestep)
        self.samples = []
        ep_start = 0
        for ep_idx in range(self.num_episodes):
            ep_end = int(self.episode_ends[ep_idx])
            ep_len = ep_end - ep_start
            for t in range(obs_horizon - 1, ep_len - pred_horizon):
                self.samples.append((ep_start, ep_end, t))
            ep_start = ep_end

        print(f"Loaded Zarr dataset: {dataset_dir}")
        print(f"  Episodes: {self.num_episodes}, Frames: {self.total_frames}")
        print(f"  Cameras: {self.camera_names}")
        print(f"  Training samples: {len(self.samples)}")

    def get_all_states(self):
        """Return all states as (N, 7) numpy array for normalization."""
        return np.array(self.states[:], dtype=np.float32)

    def get_all_actions(self):
        """Return all actions as (N, 7) numpy array for normalization."""
        return np.array(self.actions[:], dtype=np.float32)

    def _load_and_resize(self, img):
        """Resize a single BGR uint8 image to training resolution."""
        H, W = self.image_size
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        return img

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ep_start, ep_end, t = self.samples[idx]
        To = self.obs_horizon
        Tp = self.pred_horizon
        H, W = self.image_size

        # Global indices for this sample
        abs_t = ep_start + t

        # Observation: past To timesteps [t-To+1 .. t]
        obs_start = abs_t - To + 1
        obs_end = abs_t + 1
        obs_state = np.array(self.states[obs_start:obs_end], dtype=np.float32)

        # Action: next Tp timesteps [t .. t+Tp-1]
        act_start = abs_t
        act_end = abs_t + Tp
        action = np.array(self.actions[act_start:act_end], dtype=np.float32)

        # Images per camera
        result = {
            "obs_state": torch.from_numpy(obs_state),
            "action": torch.from_numpy(action),
        }

        for cam in self.camera_names:
            imgs = torch.empty((To, 3, H, W), dtype=torch.float32)
            raw_frames = self.img_arrays[cam][obs_start:obs_end]
            for j in range(To):
                img = self._load_and_resize(raw_frames[j])
                imgs[j] = torch.from_numpy(img).permute(2, 0, 1).float().div_(255.0)
            result[f"obs_img_{cam}"] = imgs

        return result
