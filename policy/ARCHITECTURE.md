# Architecture: SigLIP + DINOv2 Fused Diffusion Policy for Robotic Manipulation

## Overview

This system teaches a robot arm to pick objects by learning from human demonstrations.
A human teleoperates the robot 41 times to pick an object, and the system learns to
reproduce that behavior autonomously using camera images and joint positions.

The architecture has three main components:
1. **Vision Encoders** — understand what the cameras see
2. **Diffusion Policy** — predict what the robot should do next
3. **Action Execution** — send commands to the physical robot

---

## 1. Vision: SigLIP + DINOv2 Fused Encoder

### What it does
Takes a camera image (1280x720 pixels) and compresses it into a small vector of 512 numbers
that represent everything important about the scene — where the object is, where the gripper is,
and how they relate spatially.

### Why two encoders instead of one

**SigLIP** (Sigmoid Language-Image Pre-training, by Google)
- Full name: SigLIP Vision Transformer Base Patch-16 at 224x224 resolution
- Model: `google/siglip-base-patch16-224`
- Parameters: ~86 million
- What it learned: Trained on millions of internet image-text pairs
- What it provides: **Semantic understanding** — knows what objects are, their categories,
  their typical appearance. Understands "this is a cup" or "this is a tool."
- Output: 768-dimensional feature vector

**DINOv2** (Self-DIstillation with NO labels, version 2, by Meta/Facebook)
- Full name: DINOv2 Vision Transformer Base
- Model: `facebook/dinov2-base`
- Parameters: ~86 million
- What it learned: Trained on 142 million images using self-supervised learning (no labels)
- What it provides: **Spatial understanding** — knows where things are, their shapes, edges,
  depth relationships. Understands "the object is 3cm to the left and slightly below the gripper."
- Output: 768-dimensional feature vector

### How they fuse

```
Camera Image (224x224 RGB)
        │
        ├──→ SigLIP ──→ 768 numbers (what is this?)
        │
        └──→ DINOv2 ──→ 768 numbers (where is everything?)
                              │
                     Concatenate: 1536 numbers
                              │
                     MLP Projection: 1536 → 768 → 512 numbers
                              │
                     Final: 512-dim feature vector per camera
```

The projection MLP (Multi-Layer Perceptron) is a small neural network with two linear layers
and a GELU activation that learns to combine the semantic and spatial features into a compact
representation useful for robot control.

### Two cameras

The system uses two cameras:
- **Global camera** — fixed, overlooks the workspace from above/side
- **Wrist camera** — mounted on the robot arm, moves with the gripper

Each camera image goes through the same fused encoder independently, producing two 512-dim vectors.

### Frozen vs Finetuned

- **Frozen**: SigLIP and DINOv2 weights are fixed (not updated during training). Only the
  projection MLP learns. Fast to train (~3 seconds/epoch).
- **Finetuned**: All weights update during training. Slower (~200 seconds/epoch) but the
  encoders adapt to your specific robot setup and lighting conditions.

---

## 2. Action Prediction: Conditional Denoising Diffusion Policy

### What is a Diffusion Policy?

A diffusion policy is a method from the paper "Diffusion Policy: Visuomotor Policy Learning
via Action Diffusion" (Chi et al., RSS 2023). Instead of directly predicting one action,
it predicts a **sequence of future actions** by gradually removing noise from random data.

### How diffusion works

**Training:**
1. Take a real action sequence from a demonstration (16 future timesteps)
2. Add random noise to it (controlled amount, from a schedule of 100 noise levels)
3. Ask the neural network: "given the current camera view and robot state, predict what
   noise was added"
4. Train with MSE loss: how well did the network predict the actual noise?

**Inference (on the robot):**
1. Start with pure random noise shaped like an action sequence (16 × 7 numbers)
2. Ask the network: "what noise is in this?" (conditioned on current camera + state)
3. Remove the predicted noise slightly
4. Repeat for 10 steps (DDIM scheduler)
5. The result is a clean, predicted action sequence

### The Noise Prediction Network: Conditional 1D U-Net

```
Input: Noisy action sequence (16 timesteps × 7 joints)
       + Diffusion timestep (which noise level)
       + Observation conditioning (vision features + robot state)

Architecture:
    ┌─────────────────────────────────────────────┐
    │                                             │
    │  Timestep → Sinusoidal Embedding → MLP      │  (tells the network which noise level)
    │                            │                │
    │  Observation → Flatten → [concat]           │  (tells the network what it sees)
    │                            │                │
    │  ┌─── Down Path ────────────────────────┐   │
    │  │ Conv1d(7→256) + ResBlocks + FiLM     │   │  FiLM = Feature-wise Linear Modulation
    │  │ Conv1d(256→512) + ResBlocks + FiLM   │   │  (scales and shifts features based on
    │  │ Conv1d(512→1024) + ResBlocks + FiLM  │   │   the observation conditioning)
    │  └──────────────────────────────────────┘   │
    │                    │                        │
    │  ┌─── Middle ──────┘                        │
    │  │ ResBlock(1024) + FiLM × 2                │
    │  └─────────────────┐                        │
    │                    │                        │
    │  ┌─── Up Path ─────┘── Skip Connections ─┐  │
    │  │ Conv1d(1024→512) + ResBlocks + FiLM   │  │  Skip connections carry fine details
    │  │ Conv1d(512→256) + ResBlocks + FiLM    │  │  from the down path to the up path
    │  └───────────────────────────────────────┘  │
    │                    │                        │
    │  Final Conv1d(256→7)                        │
    │                    │                        │
    └────────────────────┘                        │
                         │                        │
Output: Predicted noise (16 timesteps × 7 joints)
```

- **Parameters**: ~95 million
- **Channel dimensions**: 256 → 512 → 1024 (down), 1024 → 512 → 256 (up)
- **Kernel size**: 5 (each convolution looks at 5 neighboring timesteps)
- **GroupNorm**: normalizes in groups of 8 channels (more stable than BatchNorm)
- **Mish activation**: smooth, non-monotonic activation function

### What the observation conditioning contains

For each timestep in the observation window (2 timesteps: current + previous):
- Global camera features: 512 numbers
- Wrist camera features: 512 numbers
- Robot joint state: 7 numbers (6 joint angles + gripper position)

Total per timestep: 512 + 512 + 7 = 1031 numbers
Total for 2 timesteps: 1031 × 2 = **2062 numbers** (the conditioning vector)

### EMA (Exponential Moving Average)

During training, a shadow copy of the U-Net weights is maintained. This copy is a slowly-updated
average of all past weight values. At inference time, this averaged model is used instead of the
final training weights — it produces more stable, less noisy predictions.

---

## 3. Action Format

### Absolute Actions (default)
The model predicts **target joint positions**:
```
action = [j1, j2, j3, j4, j5, j6, gripper]
```
- j1-j6: target angle for each joint (radians)
- gripper: 0.0 = closed, 1.0 = open

### Delta Actions (optional, `--delta` flag)
The model predicts **changes** from current position:
```
action = [Δj1, Δj2, Δj3, Δj4, Δj5, Δj6, Δgripper]
```
These are accumulated: `next_position = current_position + delta`

### Action Chunking
- **Prediction horizon**: 16 timesteps predicted at once
- **Action horizon**: 8 of those 16 are executed
- **Then replan**: observe again, predict new 16 steps, execute 8
- This overlap ensures smooth transitions between prediction chunks

---

## 4. Training Pipeline

### Data
- 41 human demonstrations recorded at 20 Hz
- Each demo: approach object → close gripper → lift
- Stored in Zarr format (compressed, chunked tensor arrays)
- Two camera streams (global + wrist) at 1280×720 pixels

### Normalization
- Joint states and actions normalized to [-1, 1] range using min-max normalization
- This helps the diffusion process work better (noise schedule assumes data in [-1, 1])

### Diffusion Schedule
- **DDPM** (Denoising Diffusion Probabilistic Models) during training
- **Squared cosine** beta schedule with 100 noise levels
- At training: random noise level sampled per batch
- At inference: **DDIM** (Denoising Diffusion Implicit Models) with 10 steps (faster)

### Optimizer
- AdamW with weight decay 1e-6
- Learning rate: 1e-4 for projection + U-Net, 1e-5 for vision encoders (if finetuned)
- Cosine annealing schedule (LR gradually decreases to 0)
- Gradient clipping at norm 1.0

---

## 5. Inference (Running on the Robot)

```
┌─────────────────────────────────────────────────────────┐
│                    20 Hz Control Loop                    │
│                                                         │
│  1. Read robot joint angles                             │
│  2. Read global + wrist camera frames                   │
│  3. If action queue is low (≤2 remaining):              │
│     a. Run SigLIP + DINOv2 on both camera frames        │
│     b. Build conditioning vector (vision + state)        │
│     c. Run DDIM denoising (10 steps) → 16 future actions│
│     d. Queue first 8 actions for execution               │
│  4. Pop next action from queue                           │
│  5. Send to ReplayController                             │
│                                                         │
│  ReplayController (50 Hz background thread):             │
│  - Linearly interpolates between consecutive targets     │
│  - Sends smooth joint commands to robot firmware         │
│  - Same controller used for demo replay (proven smooth)  │
│                                                         │
│  Auto-stop: when gripper opens then closes for 1 second, │
│  pick is considered complete → robot goes home            │
└─────────────────────────────────────────────────────────┘
```

---

## 6. Summary Table

| Component | Model | Parameters | Role |
|-----------|-------|-----------|------|
| SigLIP | google/siglip-base-patch16-224 | 86M | Semantic visual features |
| DINOv2 | facebook/dinov2-base | 86M | Spatial visual features |
| Projection MLP | Custom (1536→768→512) | 1.6M | Fuses vision features |
| 1D U-Net | Custom (256/512/1024) | 95M | Predicts action sequences |
| **Total** | | **269M** | |

| Setting | Value |
|---------|-------|
| Observation horizon | 2 timesteps |
| Prediction horizon | 16 timesteps |
| Action horizon | 8 timesteps (execute) |
| Training noise steps | 100 (DDPM) |
| Inference noise steps | 10 (DDIM) |
| Recording FPS | 20 Hz |
| Control rate | 20 Hz |
| Command rate | 50 Hz (interpolated) |

---

## References

1. Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion", RSS 2023
2. Zhai et al., "Sigmoid Loss for Language Image Pre-Training" (SigLIP), ICCV 2023
3. Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision", TMLR 2024
4. Ho et al., "Denoising Diffusion Probabilistic Models" (DDPM), NeurIPS 2020
5. Song et al., "Denoising Diffusion Implicit Models" (DDIM), ICLR 2021
