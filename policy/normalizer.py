"""Min-max normalizer to [-1, 1] with save/load support."""

import numpy as np
import torch


class MinMaxNormalizer:
    def __init__(self):
        self.data_min = None
        self.data_max = None
        self.scale = None
        self.offset = None

    def fit(self, data):
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        data = np.asarray(data, dtype=np.float32)
        self.data_min = data.min(axis=0)
        self.data_max = data.max(axis=0)
        data_range = self.data_max - self.data_min
        data_range[data_range < 1e-6] = 2.0
        self.scale = 2.0 / data_range
        self.offset = -1.0 - self.data_min * self.scale

    def normalize(self, x):
        if isinstance(x, np.ndarray):
            return x * self.scale + self.offset
        s = torch.as_tensor(self.scale, dtype=x.dtype, device=x.device)
        o = torch.as_tensor(self.offset, dtype=x.dtype, device=x.device)
        return x * s + o

    def unnormalize(self, x):
        if isinstance(x, np.ndarray):
            return (x - self.offset) / self.scale
        s = torch.as_tensor(self.scale, dtype=x.dtype, device=x.device)
        o = torch.as_tensor(self.offset, dtype=x.dtype, device=x.device)
        return (x - o) / s

    def state_dict(self):
        return {"data_min": self.data_min, "data_max": self.data_max,
                "scale": self.scale, "offset": self.offset}

    def load_state_dict(self, d):
        self.data_min = d["data_min"]
        self.data_max = d["data_max"]
        self.scale = d["scale"]
        self.offset = d["offset"]
