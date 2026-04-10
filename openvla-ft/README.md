# OpenVLA Fine-Tuning for Piper Arm Pick-and-Place

## Overview

Fine-tune OpenVLA (7B vision-language-action model) on your recorded Zarr demonstrations.
Trains on a rented cloud GPU (RunPod A100), runs inference locally on your RTX 4090.

## What's Inside

```
openvla-ft/
├── README.md                 # This file
├── setup.sh                  # Run on cloud: installs everything + converts data
├── convert_zarr_to_rlds.py   # Converts your Zarr data to RLDS format
├── register_dataset.py       # Registers your dataset with OpenVLA
├── run_lora.sh               # LoRA finetuning (1x A100, ~$15-22)
├── run_oft.sh                # OFT finetuning (4-8x A100, ~$150-500)
├── export_model.py           # Merge LoRA weights for download
├── inference.py              # Local inference on RTX 4090 + Piper arm
└── data/                     # PUT YOUR ZARR DATA HERE before uploading
```

## Step-by-Step

### 1. Prepare Data (on your PC)

Copy your Zarr dataset into the data/ folder:

```bash
cp -r data/split/pick_trimmed openvla-ft/data/pick_trimmed
```

### 2. Rent a GPU (RunPod)

1. Go to https://www.runpod.io
2. Create account, add $25 credits
3. Deploy a pod:
   - GPU: A100 80GB (for LoRA) or 4x A100 80GB (for OFT)
   - Template: RunPod PyTorch 2.2
   - Disk: 100GB (model is ~14GB + data)
4. Connect via SSH or web terminal

### 3. Upload This Folder to RunPod

From your PC:
```bash
# Using RunPod's built-in upload, or:
scp -r openvla-ft/ root@<POD_IP>:/workspace/
```

### 4. Run Setup (on RunPod)

```bash
cd /workspace/openvla-ft
bash setup.sh
```

This will:
- Install all dependencies (PyTorch, transformers, flash-attn, etc.)
- Clone the OpenVLA repository
- Convert your Zarr data to RLDS format
- Register your custom dataset with OpenVLA

Takes ~10-15 minutes.

### 5. Run Training (on RunPod)

For LoRA (recommended first try):
```bash
bash run_lora.sh
```
- 1x A100 80GB
- ~10-15 hours
- Cost: ~$15-22

For OFT (better results, more expensive):
```bash
bash run_oft.sh
```
- 4-8x A100 80GB
- ~1-2 days
- Cost: ~$150-500

### 6. Export and Download (on RunPod)

After training completes:
```bash
python export_model.py
```

This merges LoRA weights into the base model. Download the result:

From your PC:
```bash
scp -r root@<POD_IP>:/workspace/openvla-ft/output/merged_model ./checkpoints/openvla_pick/
```

### 7. Run on Robot (on your PC)

```bash
python openvla-ft/inference.py \
  --model_path ./checkpoints/openvla_pick \
  --can can0 --global-cam 0 --wrist-cam 2 \
  --instruction "pick the object from the table"
```

## Data Format

Your Zarr dataset should have:
```
dataset.zarr/
├── data/
│   ├── state       (N, 7)          float32  — 6 joints + gripper
│   ├── action      (N, 7)          float32  — next joint targets
│   ├── img_global  (N, 720, 1280, 3) uint8  — global camera
│   ├── img_wrist   (N, 720, 1280, 3) uint8  — wrist camera
│   └── ...
└── meta/
    └── episode_ends (E,)           int64
```

## Notes

- Training uses the OpenVLA-7B base model from HuggingFace (auto-downloaded)
- LoRA trains only 1.4% of parameters — fast and memory efficient
- Inference on RTX 4090 uses 4-bit quantization (~12GB VRAM)
- Action format: 7-DoF joint control (6 joints + gripper)
- Language instruction is embedded — you can change it at inference time
