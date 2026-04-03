"""All diffusion policy hyperparameters in one place."""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class DiffusionPolicyConfig:
    # Dataset
    dataset_dir: str = "./data/pick"
    image_size: Tuple[int, int] = (240, 426)  # ~half of 720x1280, keeps 16:9 aspect
    camera_names: Tuple[str, ...] = ("global", "wrist")

    # Observation
    obs_horizon: int = 2
    obs_state_dim: int = 7          # j1-j6 + gripper
    num_cameras: int = 2            # global + wrist
    vision_feature_dim: int = 256   # per camera

    # Action
    pred_horizon: int = 16
    action_horizon: int = 8
    action_dim: int = 7             # j1-j6 + gripper

    # U-Net
    down_dims: Tuple[int, ...] = (256, 512, 1024)
    diffusion_step_embed_dim: int = 256
    kernel_size: int = 5
    n_groups: int = 8
    cond_predict_scale: bool = True

    # Diffusion
    num_train_timesteps: int = 100
    num_inference_steps: int = 10
    beta_schedule: str = "squaredcos_cap_v2"

    # Training
    batch_size: int = 16
    num_epochs: int = 500
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    betas: Tuple[float, float] = (0.95, 0.999)
    grad_clip_norm: float = 1.0
    num_workers: int = 0
    ema_power: float = 0.75

    # Checkpointing
    output_dir: str = "./checkpoints/pick"
    save_every_epochs: int = 50
    log_every_steps: int = 50

    # Logging
    use_wandb: bool = False
    wandb_project: str = "piper-diffusion"

    # Inference
    control_hz: float = 20.0
    can_port: str = "can0"
    camera_id: int = 0
    robot_speed: int = 50
    max_steps: int = 600
