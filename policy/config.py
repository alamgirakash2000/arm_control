"""Configuration for SigLIP + DINOv2 + Diffusion Transformer (DiT) policy.

Architecture:
    Vision: SigLIP-base + DINOv2-base → concatenate (1536-dim) → project (512-dim)
    Action: DiT-Small (12 layers, 384 dim, 6 heads) denoises action sequences
    Conditioning: cross-attention from DiT to observation features
"""


class Config:
    # ── Vision Encoders ──────────────────────────────────────────────
    siglip_model = "google/siglip-base-patch16-224"
    dinov2_model = "facebook/dinov2-base"
    siglip_dim = 768
    dinov2_dim = 768
    fused_dim = 1536        # siglip + dinov2 concat
    proj_dim = 512          # projected per camera

    # ── Images ───────────────────────────────────────────────────────
    image_size = (224, 224)
    camera_names = ("global", "wrist")
    num_cameras = 2

    # ── Observation ──────────────────────────────────────────────────
    obs_horizon = 2
    obs_state_dim = 7       # 6 joints + gripper

    # ── Action ───────────────────────────────────────────────────────
    pred_horizon = 16
    action_horizon = 8
    action_dim = 7
    use_delta = False

    # ── Diffusion Transformer (DiT-Small from ScaleDP) ───────────────
    dit_hidden_dim = 384
    dit_num_layers = 12
    dit_num_heads = 6
    dit_mlp_ratio = 4.0

    # ── Diffusion Schedule ───────────────────────────────────────────
    num_train_timesteps = 100
    num_inference_steps = 20
    beta_schedule = "squaredcos_cap_v2"

    # ── Training ─────────────────────────────────────────────────────
    batch_size = 8
    freeze_vision = False
    num_epochs = 500
    learning_rate = 1e-4
    vision_lr = 1e-5
    weight_decay = 1e-6
    betas = (0.95, 0.999)
    grad_clip_norm = 1.0
    ema_power = 0.75

    # ── Logging ──────────────────────────────────────────────────────
    log_every_steps = 50
    save_every_epochs = 50

    @property
    def obs_feature_dim(self):
        """Per-timestep observation feature: 2 cameras × proj_dim + state."""
        return self.num_cameras * self.proj_dim + self.obs_state_dim

    @property
    def global_cond_dim(self):
        """Flattened conditioning vector."""
        return self.obs_horizon * self.obs_feature_dim
