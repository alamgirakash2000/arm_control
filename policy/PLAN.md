# VLA Pipeline Plan

## Data Collection Flow

```
Record full demo (pick + place as one continuous demo)
    │
    ▼
Partition into segments with text labels:
    Segment 1: "pick the black object"    (frames 0 → gripper closes)
    Segment 2: "place on the good box"    (frames after close → release)
    │
    ▼
Train ONE model on ALL segments with text conditioning
    │
    ▼
Inference: pass text instruction → model executes that specific task
```

## What Changes From Current Pipeline
- Recording: NO change — record full pick+place as one demo
- Partitioning: auto-split + assign text labels per segment
- Dataset: adds text label per frame
- Training: SigLIP text encoder provides language conditioning to U-Net
- Inference: pass instruction string, model follows it
