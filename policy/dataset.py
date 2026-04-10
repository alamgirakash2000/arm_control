"""Zarr dataset for SigLIP+DINOv2 Diffusion Policy.

Two modes:
- Live images: returns raw images for finetuned vision encoders
- Cached features: returns pre-computed SigLIP+DINOv2 features (frozen mode, 10x faster)
"""

import cv2
import numpy as np
import zarr
import torch
from torch.utils.data import Dataset


class PiperDataset(Dataset):
    """Loads images + state/action windows from Zarr.

    For finetuned vision: returns raw images (To, 3, 224, 224).
    For frozen vision: returns cached features (To, proj_dim) after caching step.
    """

    def __init__(self, zarr_path, obs_horizon=2, pred_horizon=16, use_delta=False):
        root = zarr.open_group(zarr_path, mode="r")

        print("  Loading state/action into RAM ...", end=" ", flush=True)
        self.state = np.array(root["data/state"][:], dtype=np.float32)
        self.action = np.array(root["data/action"][:], dtype=np.float32)
        episode_ends = root["meta/episode_ends"][:]

        # Images stay on disk
        self.img_global = root["data/img_global"]
        self.img_wrist = root["data/img_wrist"]

        self.delta = self.action - self.state
        self.use_delta = use_delta
        self.use_cached = False
        self.cached_global = None
        self.cached_wrist = None
        print("done")

        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon

        self.valid_indices = []
        starts = np.concatenate([[0], episode_ends[:-1]])
        for ep_start, ep_end in zip(starts, episode_ends):
            t_min = int(ep_start) + obs_horizon - 1
            t_max = int(ep_end) - pred_horizon
            for t in range(t_min, t_max + 1):
                self.valid_indices.append(t)

        print(f"  Samples: {len(self.valid_indices)}, "
              f"action_mode={'delta' if use_delta else 'absolute'}")

    def set_cached_features(self, feat_global, feat_wrist):
        """Set pre-computed vision features. Switches dataset to cached mode."""
        self.cached_global = feat_global  # (N, proj_dim)
        self.cached_wrist = feat_wrist
        self.use_cached = True
        print(f"  Dataset switched to cached mode: "
              f"global={feat_global.shape}, wrist={feat_wrist.shape}")

    def _load_image(self, arr, idx):
        img = np.array(arr[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        return np.transpose(img, (2, 0, 1))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        t = self.valid_indices[idx]
        To = self.obs_horizon
        Tp = self.pred_horizon

        obs_state = self.state[t - To + 1: t + 1]

        if self.use_delta:
            action = self.delta[t: t + Tp]
        else:
            action = self.action[t: t + Tp]

        if self.use_cached:
            # Return pre-computed features (fast)
            feat_g = self.cached_global[t - To + 1: t + 1]  # (To, proj_dim)
            feat_w = self.cached_wrist[t - To + 1: t + 1]
            return {
                "obs_state": obs_state,
                "obs_feat_global": feat_g,
                "obs_feat_wrist": feat_w,
                "action": action,
            }
        else:
            # Return raw images (for finetuned vision)
            imgs_global = np.stack([self._load_image(self.img_global, t - To + 1 + i)
                                    for i in range(To)])
            imgs_wrist = np.stack([self._load_image(self.img_wrist, t - To + 1 + i)
                                   for i in range(To)])
            return {
                "obs_state": obs_state,
                "obs_img_global": imgs_global,
                "obs_img_wrist": imgs_wrist,
                "action": action,
            }
