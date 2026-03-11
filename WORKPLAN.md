# Tool Inspection Pipeline — Work Plan

## Goal

Build a multi-step robotic pipeline for tool quality inspection:

1. **PICK** — Pick up a tool from a known area
2. **INSPECT** — Bring the tool to a camera for visual inspection
3. **PLACE_GOOD** — Place at location A if inspection passes
4. **PLACE_BAD** — Place at location B if inspection fails

Each step is a separate diffusion policy. A Python orchestrator chains them
together and handles the branching logic (good vs bad).

---

## Architecture Overview

```
┌───────────────────────────────────────────────────────┐
│                 ORCHESTRATOR (Python)                  │
│                                                       │
│  1. Load all 4 policy checkpoints                     │
│  2. Run PICK policy                                   │
│     └─ check: camera feed → object in gripper?        │
│  3. Run INSPECT policy (bring tool to camera)         │
│     └─ call external vision system → good / bad       │
│  4. if good → run PLACE_GOOD policy                   │
│     if bad  → run PLACE_BAD policy                    │
│  5. Robot homes, loop back to step 2 for next tool    │
└───────────────────────────────────────────────────────┘
```

Each policy is a **short-horizon diffusion policy** (~2-5 seconds of motion).
The branching logic is plain Python `if/else`, not learned.

---

## Step 1: Data Collection

### What to record

Record **full continuous demos** of the entire task (pick → inspect → place)
using the existing teleop setup. This is easier than recording each phase
separately because you don't have to stop/start between phases.

**Command:**
```bash
python teleop.py --with_robot --left can0 --record ./data/tool_inspection --cameras 0
```

### How many demos

- ~30-40 demos where the tool is GOOD (pick → inspect → place at location A)
- ~30-40 demos where the tool is BAD (pick → inspect → place at location B)
- Total: 60-80 full demos

### During each demo

1. Press **R** to start recording
2. Pick up the tool
3. Bring it to the inspection camera position
4. Place it at location A (good) or location B (bad)
5. Press **S** to save

### Important tips

- Be consistent in the **semantic flow** (pick → inspect → place) but don't
  worry about exact joint positions at transitions — variation is good
- Vary the initial tool position slightly across demos for robustness
- Keep the inspection camera position consistent (always bring the tool to
  roughly the same spot in front of the camera)
- Keep placement locations A and B consistent

---

## Step 2: Partition Demos into Sub-Tasks

After collecting all demos, split each full episode into 3 sub-episodes.

### Annotation approach

Write a script that plays back each episode and lets you mark split points
with keyboard presses:

```
Episode playback:
  [frame 0 ........... frame 45] [frame 46 ........... frame 92] [frame 93 ........... frame 140]
         PICK                           INSPECT                         PLACE
  Press '1' when pick is done     Press '2' when inspect is done      (rest is place)
```

### Split criteria

| Transition | What to look for |
|---|---|
| PICK → INSPECT | Tool is firmly grasped, robot starts moving toward camera |
| INSPECT → PLACE | Tool is in front of camera, robot starts moving to placement |

### Output structure

After partitioning, you'll have 4 separate datasets:

```
data/
├── pick/              # ~60-80 episodes (from all demos)
│   ├── meta/
│   └── data/chunk-000/
├── inspect/           # ~60-80 episodes (from all demos)
│   ├── meta/
│   └── data/chunk-000/
├── place_good/        # ~30-40 episodes (from "good" demos only)
│   ├── meta/
│   └── data/chunk-000/
└── place_bad/         # ~30-40 episodes (from "bad" demos only)
    ├── meta/
    └── data/chunk-000/
```

Each sub-dataset is in the same LeRobot v2.1 parquet + mp4 format.

### Variable starting states are fine

The INSPECT sub-task will start at different joint configs across demos
because the PICK phase doesn't always end at the exact same pose. **This is
correct and desirable** — diffusion policy learns:

> "Given where I am RIGHT NOW (observation), what should I do next (action)"

It doesn't memorize trajectories from a fixed start. Variation in starting
states = better generalization at runtime.

---

## Step 3: Train Policies

Train 4 separate diffusion policy models, one per sub-task.

### Training setup (for each sub-task)

Using the LeRobot training framework:

```bash
# Example for PICK policy
python train.py \
    --dataset_dir ./data/pick \
    --policy diffusion \
    --action_type abs_joint \
    --output_dir ./checkpoints/pick \
    --num_epochs 3000 \
    --batch_size 64 \
    --action_horizon 16 \
    --observation_horizon 2 \
    --prediction_horizon 16
```

Repeat for `inspect`, `place_good`, `place_bad`.

### Key hyperparameters to tune

| Parameter | What it does | Suggested start |
|---|---|---|
| `action_horizon` | How many future steps to execute from each prediction | 8-16 |
| `prediction_horizon` | How many steps the model predicts at once | 16 |
| `observation_horizon` | How many past observations to condition on | 2 |
| `action_type` | Which action representation to use | `abs_joint` |
| `num_epochs` | Training iterations | 3000+ |

### Action representation choice

- **`abs_joint`** — Recommended for starting. Most stable, no drift accumulation.
- **`delta_joint`** — Can work but deltas accumulate error over time.

### What the model sees (inputs)

- `observation.state` — 7 floats (j1-j6 joint angles in rad + gripper 0-1)
- `observation.images.cam_0` — 480x640x3 camera image

### What the model predicts (outputs)

- `action.abs_joint` — Next 16 steps of 7 floats (j1-j6 + gripper)

---

## Step 4: Runtime Execution

### Orchestrator script

```python
# orchestrator.py

import time
from policy_runner import load_policy, run_policy
from robot_interface import Robot
from inspection_system import inspect_tool

# Load all policies
pick_policy    = load_policy("checkpoints/pick")
inspect_policy = load_policy("checkpoints/inspect")
place_good     = load_policy("checkpoints/place_good")
place_bad      = load_policy("checkpoints/place_bad")

# Connect robot
robot = Robot(can="can0")
robot.home()

while True:
    print("Starting cycle...")

    # 1. PICK
    run_policy(pick_policy, robot, max_steps=100)  # ~5 sec at 20Hz

    if not detect_object_in_gripper(robot.get_camera_frame()):
        print("Pick failed, retrying...")
        robot.home()
        continue

    # 2. INSPECT
    run_policy(inspect_policy, robot, max_steps=80)  # ~4 sec

    # 3. Get inspection result from external system
    result = inspect_tool(robot.get_camera_frame())  # "good" or "bad"
    print(f"Inspection result: {result}")

    # 4. PLACE
    if result == "good":
        run_policy(place_good, robot, max_steps=80)
    else:
        run_policy(place_bad, robot, max_steps=80)

    # 5. Home and repeat
    robot.open_gripper()
    robot.home()
    print("Cycle complete.\n")
```

### The run_policy function

```python
def run_policy(policy, robot, max_steps=100):
    """
    Run a diffusion policy on the robot.

    Each step:
      1. Read current observation (joints + camera)
      2. Feed to policy → get predicted action chunk (next K steps)
      3. Execute first action from the chunk
      4. Repeat at 20Hz
    """
    for step in range(max_steps):
        # Current observation
        obs = {
            "state": robot.get_joint_positions(),   # [j1..j6, gripper]
            "image": robot.get_camera_frame(),       # 480x640x3
        }

        # Policy predicts next action chunk (e.g. 16 future steps)
        action_chunk = policy.predict(obs)

        # Execute first action from the chunk
        robot.send_joint_command(action_chunk[0])

        # Optional: early stop if converged (robot stopped moving)
        if is_converged(action_chunk):
            break

        time.sleep(1.0 / 20)  # 20 Hz control loop
```

### How does each policy know what to do?

It doesn't need a command or instruction. Each policy was trained on
**different data**:

- `pick_policy` only saw pick demos → only knows how to pick
- `inspect_policy` only saw inspect demos → only knows how to bring tool to camera
- `place_good` only saw place-at-A demos → only knows how to go to location A
- `place_bad` only saw place-at-B demos → only knows how to go to location B

### How does each policy know when to stop?

Three options (simplest first):

1. **Fixed horizon** — Run for N steps. You know roughly how long each phase
   takes from your demos. Start here.
2. **Convergence detection** — Stop when predicted actions have near-zero
   deltas (the policy thinks it's done moving).
3. **Learned done signal** — Add a `done` prediction head to the model.
   More complex, save for later.

### Transition safety

When the orchestrator switches from one policy to the next, the ending state
of policy N must be **within the distribution** of starting states that
policy N+1 was trained on. Since you record full continuous demos and split
them at natural transition points, this is guaranteed.

---

## Summary Checklist

- [ ] Collect 30-40 "good" full demos (pick → inspect → place at A)
- [ ] Collect 30-40 "bad" full demos (pick → inspect → place at B)
- [ ] Write partition script to split demos at transition points
- [ ] Split into 4 datasets: pick, inspect, place_good, place_bad
- [ ] Train 4 diffusion policies (one per sub-task)
- [ ] Validate each policy individually (replay on robot)
- [ ] Write orchestrator script
- [ ] Integrate external inspection system
- [ ] Test full pipeline end-to-end

---

## Reference Papers

- [Sequential Dexterity](https://arxiv.org/abs/2309.00987) — Chaining
  dexterous policies for long-horizon manipulation (CoRL 2023)
- [Diffusion Policy](https://arxiv.org/abs/2303.04137) — Visuomotor policy
  learning via action diffusion (IJRR 2025)
- [Diffusion-VLA](https://arxiv.org/abs/2412.03293) — Autoregressive
  reasoning + diffusion policies, tested on factory sorting (ICML 2025)
- [Generative Skill Chaining](https://www.researchgate.net/publication/390175368) —
  Long-horizon skill planning with diffusion models
