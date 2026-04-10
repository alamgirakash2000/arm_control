# SigLIP + DINOv2 Fused Diffusion Policy

## Architecture

```
Camera Frame (720x1280)
    │
    ├─→ SigLIP-base (ViT-B/16, 86M) ─→ 768-dim (semantic)
    │
    └─→ DINOv2-base (ViT-B/14, 86M) ─→ 768-dim (spatial)
                                            │
                                    concat: 1536-dim
                                            │
                                    project: 512-dim per camera
                                            │
            ┌───────────────────────────────┘
            │
    [global_512, wrist_512, state_7] × obs_horizon(2) = 2062-dim conditioning
            │
    ┌───────┴───────┐
    │ 1D U-Net      │  ← diffusion timestep + conditioning
    │ (81M params)  │
    │ pred_horizon=16│
    └───────┬───────┘
            │
    16 future actions (7-dim: 6 joints + gripper)
    execute first 8, then replan
```

## Key Features

- **Both encoders finetuned** (not frozen) — learns robot-specific spatial features
- **Delta actions** — predicts velocity-like changes, smoother than absolute positions
- **SigLIP** — semantic understanding (what objects are)
- **DINOv2** — spatial reasoning (where objects are)
- **Future-ready** — SigLIP accepts text prompts for language-conditioned tasks

## Files

```
policy/
├── config.py       — hyperparameters
├── dataset.py      — Zarr dataloader with image preprocessing
├── model.py        — FusedVisionEncoder + ConditionalUnet1D + EMA
├── normalizer.py   — min-max normalization
├── train.py        — training script
├── infer.py        — robot inference
└── README.md       — this file
```

## Training

```bash
# Uses arm_control conda environment
# Install deps first: pip install transformers diffusers

python policy/train.py \
    --dataset_dir ./data/split/pick_trimmed \
    --output_dir ./checkpoints/pick_fused \
    --epochs 500

# Resume
python policy/train.py \
    --dataset_dir ./data/split/pick_trimmed \
    --output_dir ./checkpoints/pick_fused \
    --resume latest --epochs 500
```

## Inference

```bash
python policy/infer.py \
    --checkpoint ./checkpoints/pick_fused/best.pt \
    --can can0 --global-cam 0 --wrist-cam 2
```

## Hardware Requirements

- **Training**: RTX 4090 24GB (batch_size=16, ~20GB VRAM)
- **Inference**: RTX 4090 24GB (~8GB VRAM)
- **Training time**: ~30-60 min/epoch depending on dataset size

## Data Format

Same Zarr format as recording pipeline:
```
dataset.zarr/
├── data/
│   ├── state       (N, 7)  — 6 joints + gripper
│   ├── action      (N, 7)  — next joint targets
│   ├── img_global  (N, H, W, 3) — global camera BGR
│   ├── img_wrist   (N, H, W, 3) — wrist camera BGR
│   ├── eef_pos     (N, 3)
│   └── eef_euler   (N, 3)
└── meta/
    └── episode_ends (E,)
```
