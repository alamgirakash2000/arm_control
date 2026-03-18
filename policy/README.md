# Diffusion Policy Training & Inference

Train and deploy diffusion policies for the Piper robot arm sub-tasks (pick, inspect, place).

## Files

| File | Purpose |
|------|---------|
| `train.py` | Training script — main entry point |
| `infer.py` | Run a trained checkpoint on the real robot |
| `model.py` | Model architecture (ResNet18 + ConditionalUnet1D) |
| `dataset.py` | Loads our parquet + mp4 datasets |
| `normalizer.py` | Min-max normalizer for state/action |
| `config.py` | All hyperparameters in one place |

## Training

```bash
# Recommended: watchdog supervisor with auto-resume on crashes/stalls
bash policy/train_loop.sh ./data/pick ./checkpoints/pick 500 20

# Direct trainer entry point
python policy/train.py --dataset_dir ./data/pick --output_dir ./checkpoints/pick

# Train inspect policy
python policy/train.py --dataset_dir ./data/inspect --output_dir ./checkpoints/inspect

# Train place_good policy
python policy/train.py --dataset_dir ./data/place_good --output_dir ./checkpoints/place_good
```

### Options

```bash
--epochs 500          # number of training epochs (default: 500)
--batch_size 64       # batch size (default: 64)
--lr 1e-4             # learning rate (default: 1e-4)
--wandb               # enable wandb logging
--resume latest       # resume from latest checkpoint
--resume path/to.pt   # resume from specific checkpoint
```

`train_loop.sh` accepts extra trainer args after the first 4 positional args:

```bash
bash policy/train_loop.sh ./data/pick ./checkpoints/pick 500 20 --batch_size 16
```

Useful watchdog env vars:

```bash
STALL_TIMEOUT_SEC=1200 bash policy/train_loop.sh ./data/pick ./checkpoints/pick 500 20
```

### Checkpoints

Saved in `--output_dir`:
- `best.pt` — lowest training loss
- `latest.pt` — most recent epoch (for resuming)
- `epoch_0050.pt`, `epoch_0100.pt`, ... — every 50 epochs
- `training_status.json` — heartbeat/progress file used by the watchdog

## Testing on Robot

```bash
# Test a checkpoint on the real robot
python policy/infer.py --checkpoint ./checkpoints/pick/best.pt --can can0 --camera 0

# Test a specific epoch
python policy/infer.py --checkpoint ./checkpoints/pick/epoch_0250.pt --can can0 --camera 0

# Dry run (camera only, no robot connected)
python policy/infer.py --checkpoint ./checkpoints/pick/best.pt --camera 0 --no_robot
```

### Options

```bash
--max_steps 600       # max control steps at 20Hz (default: 600 = 30 seconds)
--speed 50            # robot speed 1-100 (default: 50)
--can can0            # CAN port (default: can0)
--camera 0            # USB camera device ID (default: 0)
```

Press **Q** in the display window to stop early.

## Architecture

- **Vision**: ResNet18 + SpatialSoftmax → 256-dim features (input: 240x320, full FOV)
- **Noise network**: ConditionalUnet1D with FiLM conditioning
- **Diffusion**: DDPM 100 steps (training), DDIM 10 steps (inference)
- **Action**: predicts 16 future steps, executes 8 before re-predicting
- **Total**: ~84M parameters

## Data Format

Expects datasets in our LeRobot v2.1 format:
```
data/pick/
├── data/chunk-000/episode_XXXXXX.parquet
├── videos/chunk-000/observation.images.cam_0/episode_XXXXXX.mp4
└── meta/info.json, episodes.jsonl
```

Uses `observation.state` (7-dim) and `action.abs_joint` (7-dim) from parquet,
plus camera frames from mp4 video.
