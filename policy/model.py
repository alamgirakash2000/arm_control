"""SigLIP + DINOv2 fused vision encoder + Diffusion Transformer (DiT).

Architecture (ScaleDP-style):
    FusedVisionEncoder: SigLIP CLS + DINOv2 CLS → concat (1536) → project (512)
    DiffusionTransformer: DiT-Small (12 layers, 384 dim, 6 heads) for action denoising
    EMAModel: exponential moving average for stable inference

Key DiT features vs U-Net:
    - AdaLN (Adaptive Layer Norm) for timestep conditioning
    - Cross-attention for observation conditioning (not FiLM)
    - Non-causal self-attention across action timesteps
"""

import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Fused Vision Encoder ────────────────────────────────────────────────────

class FusedVisionEncoder(nn.Module):
    """SigLIP + DINOv2 fused into a single feature vector."""

    def __init__(self, siglip_model_name, dinov2_model_name,
                 fused_dim=1536, proj_dim=512, freeze=False):
        super().__init__()
        from transformers import SiglipVisionModel, Dinov2Model

        self.siglip = SiglipVisionModel.from_pretrained(siglip_model_name)
        self.dinov2 = Dinov2Model.from_pretrained(dinov2_model_name)
        self.frozen = freeze

        if freeze:
            for p in self.siglip.parameters():
                p.requires_grad = False
            for p in self.dinov2.parameters():
                p.requires_grad = False
            self.siglip.eval()
            self.dinov2.eval()

        self.projection = nn.Sequential(
            nn.Linear(fused_dim, 768),
            nn.GELU(),
            nn.Linear(768, proj_dim),
        )

        self.register_buffer("siglip_mean",
                             torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
        self.register_buffer("siglip_std",
                             torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
        self.register_buffer("dino_mean",
                             torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("dino_std",
                             torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, images):
        img_siglip = (images - self.siglip_mean) / self.siglip_std
        img_dino = (images - self.dino_mean) / self.dino_std

        if self.frozen:
            with torch.no_grad():
                feat_siglip = self.siglip(pixel_values=img_siglip).pooler_output
                feat_dino = self.dinov2(img_dino).pooler_output
        else:
            feat_siglip = self.siglip(pixel_values=img_siglip).pooler_output
            feat_dino = self.dinov2(img_dino).pooler_output

        fused = torch.cat([feat_siglip, feat_dino], dim=-1)
        return self.projection(fused)


# ── Diffusion Transformer (DiT) — ScaleDP-style ────────────────────────────

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=x.device) * -emb)
        emb = x[:, None].float() * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class AdaLN(nn.Module):
    """Adaptive Layer Norm — conditions layer norm with timestep + obs embedding.

    Instead of FiLM (scale+bias on features), AdaLN modulates the
    layer norm parameters themselves. This is the key innovation from DiT/ScaleDP.
    """

    def __init__(self, hidden_dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        # Predict scale (gamma) and shift (beta) from conditioning
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, hidden_dim * 2),
        )

    def forward(self, x, cond):
        """
        x: (B, T, D)
        cond: (B, cond_dim)
        """
        gamma_beta = self.proj(cond).unsqueeze(1)  # (B, 1, 2*D)
        gamma, beta = gamma_beta.chunk(2, dim=-1)  # each (B, 1, D)
        return self.norm(x) * (1 + gamma) + beta


class DiTBlock(nn.Module):
    """Single Diffusion Transformer block.

    Components:
        1. AdaLN + Non-causal Self-Attention (action tokens attend to all others)
        2. Cross-Attention to observation conditioning
        3. AdaLN + Feed-Forward Network
    """

    def __init__(self, hidden_dim, num_heads, cond_dim, obs_dim, mlp_ratio=4.0):
        super().__init__()

        # Self-attention with AdaLN
        self.adaln_sa = AdaLN(hidden_dim, cond_dim)
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True)

        # Cross-attention to observation
        self.norm_cross = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True,
            kdim=obs_dim, vdim=obs_dim)

        # Feed-forward with AdaLN
        self.adaln_ff = AdaLN(hidden_dim, cond_dim)
        mlp_dim = int(hidden_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, hidden_dim),
        )

    def forward(self, x, cond, obs_tokens):
        """
        x: (B, T, D) — noisy action tokens
        cond: (B, cond_dim) — timestep + global conditioning
        obs_tokens: (B, N_obs, obs_dim) — observation token sequence
        """
        # Self-attention (non-causal — all action tokens see each other)
        h = self.adaln_sa(x, cond)
        h, _ = self.self_attn(h, h, h)
        x = x + h

        # Cross-attention to observation
        h = self.norm_cross(x)
        h, _ = self.cross_attn(h, obs_tokens, obs_tokens)
        x = x + h

        # Feed-forward
        h = self.adaln_ff(x, cond)
        h = self.ffn(h)
        x = x + h

        return x


class DiffusionTransformer(nn.Module):
    """Diffusion Transformer for action sequence denoising (ScaleDP-style).

    Replaces the 1D U-Net. Key advantages:
    - Cross-attention directly attends to observation features (stronger than FiLM)
    - Non-causal self-attention reduces compounding errors across action horizon
    - AdaLN provides stable conditioning of timestep information

    Configs (from ScaleDP paper):
        Tiny:  8 layers, 256 dim, 4 heads, ~10M
        Small: 12 layers, 384 dim, 6 heads, ~33M
        Base:  12 layers, 768 dim, 12 heads, ~130M
    """

    def __init__(self, action_dim=7, pred_horizon=16, hidden_dim=384,
                 num_layers=12, num_heads=6, global_cond_dim=None,
                 obs_cond_dim=None, mlp_ratio=4.0):
        super().__init__()
        self.action_dim = action_dim
        self.pred_horizon = pred_horizon
        self.hidden_dim = hidden_dim

        # Diffusion timestep embedding
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        # Condition projection (global_cond → hidden_dim for AdaLN)
        cond_dim = hidden_dim
        if global_cond_dim is not None:
            self.cond_proj = nn.Linear(global_cond_dim, hidden_dim)
        else:
            self.cond_proj = None
        # Total AdaLN conditioning = timestep_embed + projected_global_cond
        adaln_cond_dim = hidden_dim + hidden_dim if global_cond_dim else hidden_dim

        # Observation projection for cross-attention keys/values
        obs_dim = hidden_dim
        if obs_cond_dim is not None and obs_cond_dim != hidden_dim:
            self.obs_proj = nn.Linear(obs_cond_dim, hidden_dim)
        else:
            self.obs_proj = None
            obs_dim = obs_cond_dim or hidden_dim

        # Action token embedding: project action_dim → hidden_dim
        self.action_embed = nn.Linear(action_dim, hidden_dim)

        # Learnable positional embedding for action sequence
        self.pos_embed = nn.Parameter(
            torch.randn(1, pred_horizon, hidden_dim) * 0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, adaln_cond_dim, obs_dim, mlp_ratio)
            for _ in range(num_layers)
        ])

        # Final norm + projection back to action_dim
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.final_proj = nn.Linear(hidden_dim, action_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Zero-init the final projection (diffusion convention)
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)

    def forward(self, sample, timestep, global_cond=None, obs_tokens=None):
        """
        Args:
            sample: (B, T, action_dim) — noisy action sequence
            timestep: (B,) or scalar — diffusion timestep
            global_cond: (B, global_cond_dim) — flattened observation conditioning
            obs_tokens: (B, N, obs_feat_dim) — observation token sequence for cross-attn
                        If None, constructs from global_cond by reshaping
        Returns:
            (B, T, action_dim) — predicted noise
        """
        B = sample.shape[0]

        # Embed timestep
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=sample.device)
        elif len(timestep.shape) == 0:
            timestep = timestep[None].to(sample.device)
        timestep = timestep.expand(B)
        t_emb = self.time_embed(timestep)  # (B, hidden_dim)

        # Build AdaLN conditioning: [timestep_emb, projected_global_cond]
        if self.cond_proj is not None and global_cond is not None:
            cond = torch.cat([t_emb, self.cond_proj(global_cond)], dim=-1)
        else:
            cond = t_emb

        # Build observation tokens for cross-attention
        if obs_tokens is None and global_cond is not None:
            # Reshape global_cond into token sequence
            obs_tokens = global_cond.unsqueeze(1)  # (B, 1, cond_dim)
        if self.obs_proj is not None and obs_tokens is not None:
            obs_tokens = self.obs_proj(obs_tokens)

        # Embed noisy actions into hidden_dim
        x = self.action_embed(sample) + self.pos_embed[:, :sample.shape[1], :]

        # Transformer blocks
        for block in self.blocks:
            x = block(x, cond, obs_tokens)

        # Final projection
        x = self.final_norm(x)
        x = self.final_proj(x)

        return x


# ── EMA Model ───────────────────────────────────────────────────────────────

class EMAModel:
    def __init__(self, model, power=0.75):
        self.averaged_model = copy.deepcopy(model)
        self.averaged_model.eval()
        self.power = power
        self.step = 0
        self._pairs = None

    def _refresh(self, model):
        self._pairs = tuple(
            (ep, mp) for (_, ep), (_, mp)
            in zip(self.averaged_model.named_parameters(),
                   model.named_parameters()))

    @torch.no_grad()
    def update(self, model):
        if self._pairs is None:
            self._refresh(model)
        self.step += 1
        decay = 1 - (1 + self.step) ** (-self.power)
        for ema_p, model_p in self._pairs:
            ema_p.data.mul_(decay).add_(model_p.data, alpha=1 - decay)

    def state_dict(self):
        return {"averaged_model": self.averaged_model.state_dict(),
                "step": self.step}

    def load_state_dict(self, d):
        self.averaged_model.load_state_dict(d["averaged_model"])
        self.step = d["step"]
