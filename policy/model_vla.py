"""VLA Model — SigLIP (vision+text) + DINOv2 (spatial) + Diffusion U-Net.

Adds text conditioning from SigLIP's text encoder to the existing
FusedVisionEncoder + ConditionalUnet1D architecture.

Architecture:
    [SigLIP_image + DINOv2_image + SigLIP_text + state] → U-Net → actions
"""

import math
import copy

import torch
import torch.nn as nn

# Import the base components from model.py
from model import (
    ConditionalUnet1D,
    EMAModel,
    SinusoidalPosEmb,
    Downsample1d,
    Upsample1d,
    Conv1dBlock,
    ConditionalResidualBlock1D,
)


class VLAVisionEncoder(nn.Module):
    """SigLIP (vision+text) + DINOv2 fused encoder with text conditioning.

    Vision: SigLIP_image + DINOv2_image → concat → project → 512-dim
    Text:   SigLIP_text → project → 256-dim
    Total conditioning per camera: 512 (vision) + 256 (text, shared) + 7 (state)
    """

    def __init__(self, siglip_model_name, dinov2_model_name,
                 fused_dim=1536, proj_dim=512, text_proj_dim=256, freeze=False):
        super().__init__()
        from transformers import SiglipVisionModel, SiglipTextModel, Dinov2Model, AutoTokenizer

        # Vision encoders
        self.siglip_vision = SiglipVisionModel.from_pretrained(siglip_model_name)
        self.dinov2 = Dinov2Model.from_pretrained(dinov2_model_name)

        # Text encoder (always frozen — pretrained text understanding is good enough)
        self.siglip_text = SiglipTextModel.from_pretrained(siglip_model_name)
        for p in self.siglip_text.parameters():
            p.requires_grad = False
        self.siglip_text.eval()

        # Tokenizer for text
        self.tokenizer = AutoTokenizer.from_pretrained(siglip_model_name)

        self.frozen = freeze
        if freeze:
            for p in self.siglip_vision.parameters():
                p.requires_grad = False
            for p in self.dinov2.parameters():
                p.requires_grad = False
            self.siglip_vision.eval()
            self.dinov2.eval()
        else:
            self.siglip_vision.gradient_checkpointing_enable()
            self.dinov2.gradient_checkpointing_enable()

        # Vision projection
        self.vision_projection = nn.Sequential(
            nn.Linear(fused_dim, 768),
            nn.GELU(),
            nn.Linear(768, proj_dim),
        )

        # Text projection
        self.text_projection = nn.Sequential(
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Linear(512, text_proj_dim),
        )

        self.proj_dim = proj_dim
        self.text_proj_dim = text_proj_dim

        # SigLIP normalization
        self.register_buffer("siglip_mean",
                             torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
        self.register_buffer("siglip_std",
                             torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
        # DINOv2 normalization
        self.register_buffer("dino_mean",
                             torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("dino_std",
                             torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def encode_text(self, text_list, device):
        """Encode list of text strings → (B, text_proj_dim)."""
        tokens = self.tokenizer(text_list, padding=True, truncation=True,
                                max_length=64, return_tensors="pt").to(device)
        with torch.no_grad():
            text_feat = self.siglip_text(**tokens).pooler_output  # (B, 768)
        return self.text_projection(text_feat)  # (B, text_proj_dim)

    def encode_vision(self, images):
        """Encode images → (B, proj_dim)."""
        img_siglip = (images - self.siglip_mean) / self.siglip_std
        img_dino = (images - self.dino_mean) / self.dino_std

        if self.frozen:
            with torch.no_grad():
                feat_siglip = self.siglip_vision(pixel_values=img_siglip).pooler_output
                feat_dino = self.dinov2(img_dino).pooler_output
        else:
            feat_siglip = self.siglip_vision(pixel_values=img_siglip).pooler_output
            feat_dino = self.dinov2(img_dino).pooler_output

        fused = torch.cat([feat_siglip, feat_dino], dim=-1)
        return self.vision_projection(fused)
