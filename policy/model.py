"""Diffusion Policy model components.

Architecture from Chi et al. "Diffusion Policy" (2023):
- SpatialSoftmax + ResNet18 vision encoder
- ConditionalUnet1D noise prediction network
- EMA model wrapper
"""

import math
import copy
from typing import Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import einops
from einops.layers.torch import Rearrange


# ============================================================
# Building blocks
# ============================================================

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """Conv1d -> GroupNorm -> Mish"""

    def __init__(self, in_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    """Residual block with FiLM conditioning."""

    def __init__(self, in_channels, out_channels, cond_dim,
                 kernel_size=3, n_groups=8, cond_predict_scale=False):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups),
        ])

        cond_channels = out_channels * 2 if cond_predict_scale else out_channels
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            Rearrange("batch t -> batch t 1"),
        )
        self.residual_conv = (nn.Conv1d(in_channels, out_channels, 1)
                              if in_channels != out_channels else nn.Identity())

    def forward(self, x, cond):
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:, 0, ...]
            bias = embed[:, 1, ...]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


# ============================================================
# ConditionalUnet1D — noise prediction network
# ============================================================

class ConditionalUnet1D(nn.Module):
    """1D temporal U-Net for diffusion noise prediction.

    Adapted from Chi et al. "Diffusion Policy" reference implementation.
    """

    def __init__(self, input_dim, global_cond_dim=None,
                 diffusion_step_embed_dim=256, down_dims=(256, 512, 1024),
                 kernel_size=5, n_groups=8, cond_predict_scale=True):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        # Diffusion timestep encoder
        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )

        cond_dim = dsed
        if global_cond_dim is not None:
            cond_dim += global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        # Down path
        self.down_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= len(in_out) - 1
            self.down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_in, dim_out, cond_dim,
                                           kernel_size, n_groups, cond_predict_scale),
                ConditionalResidualBlock1D(dim_out, dim_out, cond_dim,
                                           kernel_size, n_groups, cond_predict_scale),
                Downsample1d(dim_out) if not is_last else nn.Identity(),
            ]))

        # Middle
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim,
                                       kernel_size, n_groups, cond_predict_scale),
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim,
                                       kernel_size, n_groups, cond_predict_scale),
        ])

        # Up path
        self.up_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= len(in_out) - 1
            self.up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_out * 2, dim_in, cond_dim,
                                           kernel_size, n_groups, cond_predict_scale),
                ConditionalResidualBlock1D(dim_in, dim_in, cond_dim,
                                           kernel_size, n_groups, cond_predict_scale),
                Upsample1d(dim_in) if not is_last else nn.Identity(),
            ]))

        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

    def forward(self, sample, timestep, global_cond=None):
        """
        Args:
            sample: (B, T, input_dim) — noisy action sequence
            timestep: (B,) or scalar — diffusion timestep
            global_cond: (B, global_cond_dim) — observation conditioning
        Returns:
            (B, T, input_dim) — predicted noise
        """
        # (B, T, D) -> (B, D, T) for Conv1d
        x = einops.rearrange(sample, "b t d -> b d t")

        # Encode timestep
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long,
                                    device=sample.device)
        elif len(timestep.shape) == 0:
            timestep = timestep[None].to(sample.device)
        timestep = timestep.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timestep)
        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], dim=-1)

        # Down
        h = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        # Middle
        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        # Up
        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        x = einops.rearrange(x, "b d t -> b t d")
        return x


# ============================================================
# Vision Encoder — ResNet18 + SpatialSoftmax
# ============================================================

class SpatialSoftmax(nn.Module):
    """Compute expected (x, y) coordinates for each feature channel."""

    def __init__(self, height, width, num_channels):
        super().__init__()
        # Create coordinate grids
        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1, 1, height),
            torch.linspace(-1, 1, width),
            indexing="ij")
        self.register_buffer("pos_x", pos_x.reshape(-1))  # (H*W,)
        self.register_buffer("pos_y", pos_y.reshape(-1))

    def forward(self, feature_map):
        """
        Args:
            feature_map: (B, C, H, W)
        Returns:
            (B, C*2) — expected (x, y) per channel
        """
        B, C, H, W = feature_map.shape
        flat = feature_map.reshape(B, C, -1)       # (B, C, H*W)
        softmax = F.softmax(flat, dim=-1)           # (B, C, H*W)
        exp_x = (softmax * self.pos_x).sum(dim=-1)  # (B, C)
        exp_y = (softmax * self.pos_y).sum(dim=-1)
        return torch.cat([exp_x, exp_y], dim=-1)   # (B, C*2)


def _replace_bn_with_gn(module, num_groups=16):
    """Replace all BatchNorm2d layers with GroupNorm."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_ch = child.num_features
            gn = nn.GroupNorm(min(num_groups, num_ch), num_ch)
            setattr(module, name, gn)
        else:
            _replace_bn_with_gn(child, num_groups)


class VisionEncoder(nn.Module):
    """ResNet18 backbone with SpatialSoftmax for spatial feature extraction."""

    def __init__(self, out_dim=256, freeze_bn=True):
        super().__init__()
        resnet = models.resnet18(weights=None)
        _replace_bn_with_gn(resnet)

        # Remove avgpool and fc — we use SpatialSoftmax instead
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        )
        # ResNet18 layer4 output: 512 channels
        # For 240x320 input -> 8x10 feature map (preserves 3:4 aspect ratio)
        self.spatial_softmax = SpatialSoftmax(8, 10, 512)
        self.projection = nn.Linear(512 * 2, out_dim)

        # ImageNet normalization
        self.register_buffer("img_mean",
                             torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1))
        self.register_buffer("img_std",
                             torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1))

    def forward(self, images):
        """
        Args:
            images: (B, 3, H, W) float32 in [0, 1] range (240x320)
        Returns:
            (B, out_dim) visual feature vector
        """
        x = (images - self.img_mean) / self.img_std
        features = self.backbone(x)          # (B, 512, 8, 10)
        spatial = self.spatial_softmax(features)  # (B, 1024)
        return self.projection(spatial)      # (B, out_dim)


# ============================================================
# EMA Model
# ============================================================

class EMAModel:
    """Exponential Moving Average of model parameters for stable inference."""

    def __init__(self, model, power=0.75):
        self.averaged_model = copy.deepcopy(model)
        self.averaged_model.eval()
        self.power = power
        self.optimization_step = 0
        self._param_pairs = None

    def refresh_parameter_cache(self, model):
        ema_named = tuple(self.averaged_model.named_parameters())
        model_named = tuple(model.named_parameters())

        if len(ema_named) != len(model_named):
            raise ValueError(
                "EMA/model parameter count mismatch: "
                f"{len(ema_named)} vs {len(model_named)}"
            )

        param_pairs = []
        for (ema_name, ema_param), (model_name, model_param) in zip(ema_named, model_named):
            if ema_name != model_name:
                raise ValueError(
                    "EMA/model parameter name mismatch: "
                    f"{ema_name!r} vs {model_name!r}"
                )
            param_pairs.append((ema_param, model_param))
        self._param_pairs = tuple(param_pairs)

    def get_decay(self, step):
        return 1 - (1 + step) ** (-self.power)

    @torch.no_grad()
    def step(self, model):
        if self._param_pairs is None:
            self.refresh_parameter_cache(model)
        self.optimization_step += 1
        decay = self.get_decay(self.optimization_step)
        for ema_p, model_p in self._param_pairs:
            ema_p.data.mul_(decay).add_(model_p.data, alpha=1.0 - decay)
