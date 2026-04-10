"""VLA Dataset — loads images + state + actions + text labels from Zarr.

Each episode has a text instruction label. The dataset returns the text
alongside vision and state data for language-conditioned training.
"""

import json
import cv2
import numpy as np
import zarr
from torch.utils.data import Dataset


class VLADataset(Dataset):
    """Multi-task dataset with text labels per episode.

    Expects:
        dataset.zarr/ — same as before
        meta/task_labels.json — [{episode_index, task, ...}, ...]
    """

    def __init__(self, zarr_path, labels_path, obs_horizon=2, pred_horizon=16,
                 use_delta=False):
        root = zarr.open_group(zarr_path, mode="r")

        print("  Loading state/action into RAM ...", end=" ", flush=True)
        self.state = np.array(root["data/state"][:], dtype=np.float32)
        self.action = np.array(root["data/action"][:], dtype=np.float32)
        episode_ends = root["meta/episode_ends"][:]
        self.img_global = root["data/img_global"]
        self.img_wrist = root["data/img_wrist"]
        self.delta = self.action - self.state
        self.use_delta = use_delta
        print("done")

        # Load text labels
        with open(labels_path) as f:
            task_labels = json.load(f)

        # Build episode→label mapping
        self.ep_labels = {}
        for t in task_labels:
            self.ep_labels[t["episode_index"]] = t["task"]

        unique_tasks = sorted(set(self.ep_labels.values()))
        print(f"  Tasks: {unique_tasks}")
        print(f"  Episodes: {len(episode_ends)}, Labels: {len(self.ep_labels)}")

        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon

        # Build valid indices + map each index to its episode (for label lookup)
        self.valid_indices = []
        self.index_to_ep = []
        starts = np.concatenate([[0], episode_ends[:-1]])
        for ep_idx, (ep_start, ep_end) in enumerate(zip(starts, episode_ends)):
            t_min = int(ep_start) + obs_horizon - 1
            t_max = int(ep_end) - pred_horizon
            for t in range(t_min, t_max + 1):
                self.valid_indices.append(t)
                self.index_to_ep.append(ep_idx)

        print(f"  Samples: {len(self.valid_indices)}")

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
        ep_idx = self.index_to_ep[idx]
        To = self.obs_horizon
        Tp = self.pred_horizon

        obs_state = self.state[t - To + 1: t + 1]

        if self.use_delta:
            action = self.delta[t: t + Tp]
        else:
            action = self.action[t: t + Tp]

        imgs_global = np.stack([self._load_image(self.img_global, t - To + 1 + i)
                                for i in range(To)])
        imgs_wrist = np.stack([self._load_image(self.img_wrist, t - To + 1 + i)
                               for i in range(To)])

        # Text label for this episode
        text = self.ep_labels.get(ep_idx, "do the task")

        return {
            "obs_state": obs_state,
            "obs_img_global": imgs_global,
            "obs_img_wrist": imgs_wrist,
            "action": action,
            "text": text,
        }
