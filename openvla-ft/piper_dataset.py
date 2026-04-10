"""Custom PyTorch Dataset for Piper arm Zarr data — plugs directly into OpenVLA finetune.py.

Replaces the RLDS dataset. Reads from Zarr, converts to EEF delta actions,
and formats for OpenVLA's action tokenizer.
"""

import os
import numpy as np
import torch
import zarr
from PIL import Image
from torch.utils.data import Dataset

# OpenVLA uses -100 to mask non-action tokens in the loss
IGNORE_INDEX = -100


class PiperZarrDataset(Dataset):
    """Load Piper arm demos from Zarr for OpenVLA fine-tuning.

    Actions: EEF deltas [dx, dy, dz, droll, dpitch, dyaw, gripper]
    Images: global camera (primary)
    """

    def __init__(self, zarr_dir, action_tokenizer, base_tokenizer,
                 image_transform, prompt_builder_fn,
                 instruction="pick the object from the table"):
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn
        self.instruction = instruction

        # Load Zarr data into RAM
        zarr_path = os.path.join(zarr_dir, "dataset.zarr")
        root = zarr.open_group(zarr_path, mode="r")

        self.eef_pos = np.array(root["data"]["eef_pos"][:], dtype=np.float32)
        self.eef_euler = np.array(root["data"]["eef_euler"][:], dtype=np.float32)
        self.gripper = np.array(root["data"]["state"][:, 6], dtype=np.float32)
        self.img_global = root["data"]["img_global"]  # keep on disk, load per-sample
        self.episode_ends = np.array(root["meta"]["episode_ends"][:], dtype=np.int64)
        self.starts = np.concatenate([[0], self.episode_ends[:-1]])
        self.N = int(self.episode_ends[-1])

        # Precompute EEF delta actions for all frames
        self.actions = np.zeros((self.N, 7), dtype=np.float32)
        for ep_idx in range(len(self.episode_ends)):
            s, e = int(self.starts[ep_idx]), int(self.episode_ends[ep_idx])
            for t in range(s, e):
                if t < e - 1:
                    delta_pos = self.eef_pos[t + 1] - self.eef_pos[t]
                    delta_euler = self.eef_euler[t + 1] - self.eef_euler[t]
                    next_grip = self.gripper[t + 1]
                else:
                    delta_pos = np.zeros(3, dtype=np.float32)
                    delta_euler = np.zeros(3, dtype=np.float32)
                    next_grip = self.gripper[t]
                self.actions[t] = np.concatenate([delta_pos, delta_euler, [next_grip]])

        # Compute dataset statistics for action de-normalization at inference
        q01 = np.percentile(self.actions, 1, axis=0).astype(np.float32)
        q99 = np.percentile(self.actions, 99, axis=0).astype(np.float32)
        self.dataset_statistics = {
            "piper_pick_trimmed": {
                "action": {"q01": q01, "q99": q99, "mean": self.actions.mean(axis=0).astype(np.float32), "std": self.actions.std(axis=0).astype(np.float32)},
                "proprio": {"q01": np.zeros(8, dtype=np.float32), "q99": np.ones(8, dtype=np.float32)},
                "num_transitions": self.N,
                "num_trajectories": len(self.episode_ends),
            }
        }

        print(f"  PiperZarrDataset: {len(self.episode_ends)} episodes, "
              f"{self.N} frames, action_dim=7 (EEF delta)")

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # Load image (global camera only — primary for OpenVLA)
        img_bgr = np.array(self.img_global[idx])  # (H, W, 3) BGR uint8
        img_rgb = img_bgr[:, :, ::-1].copy()       # BGR -> RGB
        image = Image.fromarray(img_rgb)

        # Get action
        action = self.actions[idx]  # (7,) float32

        # Build prompt
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {self.instruction}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize
        input_ids = self.base_tokenizer(
            prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Convert to tensors
        input_ids = torch.tensor(input_ids)
        labels = torch.tensor(labels)
        pixel_values = self.image_transform(image)

        # Mask everything except action tokens in the loss
        labels[: -(len(action) + 1)] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
