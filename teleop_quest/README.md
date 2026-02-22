# Piper Dual-Arm Teleop — Meta Quest 3 (SteamVR / OpenVR)

Real-time bilateral teleoperation of two hanging Piper arms using **Meta Quest 3 Touch controllers** via SteamVR and the OpenVR Python library.



**Neck-mounted operation:** hang the Quest 3 on a lanyard so the front cameras point forward at roughly chest height. The inside-out tracking still sees your controllers held naturally in front of you. You can watch the actual robots directly — critical for safety and precise manipulation.

---

## Hardware Requirements

| Component | Notes |
|---|---|
| Meta Quest 3 | Both Touch Plus controllers required |
| 2× AgileX Piper arms | Mounted upside-down (hanging) |
| 2× USB-to-CAN adapters | One per arm |
| Windows / Linux PC | Running SteamVR, connected to CAN adapters |
| Wi-Fi or USB-C cable | For Air Link / Quest Link connection to PC |

---

## Software Requirements

```bash
# Required
pip install openvr

# Optional but recommended (local status overlay window)
pip install opencv-python

# Also needs piper_core from the sibling teleop/ folder
# (no install needed — path is added automatically by the script)
```

**SteamVR** must be installed and running. The Meta Quest app (Windows) handles the Air Link / Quest Link connection.

---

## Files

| File | Purpose |
|---|---|
| `quest_receiver.py` | **Main teleop script** — Quest controllers → dual Piper arms |
| `README.md` | This file |


## Quick Start

### 1. CAN setup (once per session)

```bash
sudo bash ../teleop/start_teleop.sh
```

### 2. Connect Quest 3 to PC

- **Air Link (wireless):** open Meta Quest app on PC → Devices → Air Link
- **Quest Link (wired):** plug USB-C cable, accept connection in headset
- Open SteamVR. Confirm both controllers show up as tracked.

### 3. Simulation mode — safe, no robot motion

```bash
python3 quest_receiver.py
```

Hold both controllers, watch the terminal and local cv2 window. IK is validated live. Nothing moves.

### 4. Robot mode

```bash
python3 quest_receiver.py --with_robot
python3 quest_receiver.py --with_robot --left can0 --right can1   # explicit ports
python3 quest_receiver.py --with_robot --speed 15 --scale 0.5     # conservative
```

---

## All CLI Options

| Flag | Default | Description |
|---|---|---|
| `--with_robot` | off | Enable real CAN commands |
| `--left IFACE` | auto | CAN interface for left arm |
| `--right IFACE` | auto | CAN interface for right arm |
| `--scale N` | `1.0` | Motion scale 0.1–2.0 |
| `--speed N` | `20` | Robot speed 1–100 % |
| `--smoothing N` | `0.4` | EMA alpha: `0`=frozen, `1`=raw (no filtering) |
| `--prediction N` | `0.01` | OpenVR pose prediction horizon (seconds) |
| `--ref-secs N` | `5.0` | Seconds to average for home pose capture |
| `--invert-gripper` | off | Fully pressed trigger = open (default = close) |

---

## Gripper Control

The **index trigger** maps linearly to gripper opening:

```
trigger 0.0  (released)  →  gripper 0%   = fully closed
trigger 1.0  (pressed)   →  gripper 100% = fully open (70mm)
```

Use `--invert-gripper` to flip this if your application prefers the opposite convention.

The gripper is deadband-gated: commands are only sent to the robot when the opening changes by more than `GRIPPER_DEADBAND` (2%) to avoid CAN bus spam.

---

## Calibration Procedure

### How the mapping works

On startup you get a countdown. Hold both controllers **exactly where you want the robot home to be** (matching the robot's `HOME_POSITION` joint configuration). Every frame thereafter:

```
robot_target = robot_home + remap(controller_current - controller_home) * scale
```

This is an **absolute position** mapping — no drift. The robot always reflects the total displacement of your hand from home.

### EMA smoothing

The `--smoothing` parameter controls exponential moving average on the deltas:
- `0.0` — completely frozen (never moves)
- `0.4` (default) — good balance of responsiveness vs. tremor rejection
- `1.0` — raw, no smoothing (may feel jittery)

The deadband (`DEADBAND_CM`, `DEADBAND_DEG`) suppresses sub-threshold motion before the EMA, so small tracking noise near zero gets zeroed first, then smoothed.

### Step 1 — Find your axis mapping

The OpenVR coordinate frame is:  **X = right, Y = up, -Z = forward**

Run simulation mode and move one controller axis at a time:

```
Controller moves FORWARD  → watch dx/dy/dz → that is VP_Z → robot X (or Y or Z)
Controller moves UP       → watch dx/dy/dz → that is VP_Y → robot ...
Controller moves RIGHT     → watch dx/dy/dz → that is VP_X → robot ...
```

Edit `AXIS_MAP_L` and `AXIS_MAP_R` at the top of `quest_receiver.py`:

```python
# Format: [(OpenVR_axis_index, sign), ...]  for [robot_x, robot_y, robot_z]
AXIS_MAP_L = [(1, 1), (0,  1), (2, -1)]   # default
AXIS_MAP_R = [(1, 1), (0, -1), (2, -1)]   # right arm: lateral mirrored
```

These defaults match `../teleop/teleop_receiver.py` (Vision Pro). Since both use the same world-frame convention (Y up, -Z forward), calibration findings transfer between scripts.

### Step 2 — Verify IK in simulation

Before enabling the robot, run simulation and move your controllers through the full expected workspace. The display should show **IK OK** (green) throughout. If you see `IK failed` or `joint limits exceeded`, reduce `--scale`.

### Step 3 — First robot run

```bash
python3 quest_receiver.py --with_robot --speed 10 --scale 0.3
```

Very low speed and scale. Verify each arm moves in the correct direction, then increase.

---

## Architecture

```
Meta Quest 3 controllers
         │  SteamVR  (Air Link / Quest Link)
         ▼
   openvr.getDeviceToAbsoluteTrackingPose()   -- poll at 90 Hz
         │
         ├─ find_controllers()     -- L/R assignment (role-based + fallback)
         ├─ poll_controller()      -- pos [3], R [3,3], trigger [0-1], status
         │
         ├─ remap()               -- OpenVR axes → robot EE axes
         ├─ deadband()            -- tremor suppression
         ├─ EMA smooth            -- alpha * raw + (1-alpha) * prev
         ├─ rotation_delta()      -- R_cur @ R_home.T → euler
         ├─ validate()            -- IK check: home + delta reachable?
         │
         ├─ [simulation]  terminal + local cv2 window  (no robot motion)
         │
         └─ [--with_robot]
              ├─ ArmCommander("left")   ← background thread, non-blocking
              └─ ArmCommander("right")  ← background thread, non-blocking
                    └─ send_cartesian(x, y, z, r, p, yaw, timeout=0.05s)
```

**Key differences vs. Vision Pro (`teleop_receiver.py`):**
- Input: `openvr` instead of `avp_stream`
- Gripper: trigger instead of pinch distance
- Display: local `cv2.imshow()` instead of streaming to the headset app
- Added: EMA smoothing for controller tracking noise
- Added: Hold-last on tracking loss (EMA state persists, doesn't snap to zero)
- Reference: SVD-based rotation average over `--ref-secs` seconds (more robust than single-frame snapshot)

---

## Neck-Mount Tips

- Use a **short, stiff lanyard** or 3D-print a chest mount that holds the headset flat against your sternum — reduces sway that could create false deltas
- The 1 cm deadband absorbs small headset wobble
- If you see sporadic jumps in the display, increase `--smoothing` to 0.6–0.7
- The headset cameras need line of sight to the controllers — keep your hands in front of your body, not behind or far to the sides

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Could not initialize OpenVR` | Start SteamVR first; wake the headset |
| Controllers not found | Check SteamVR room setup; wiggle controllers to wake them |
| Roles show as UNKNOWN | Fallback assignment kicks in; check SteamVR controller binding settings |
| Arms move wrong axis | Calibrate `AXIS_MAP_L` / `AXIS_MAP_R` (see above) |
| Arms move opposite direction | Negate the sign in the relevant axis map entry |
| `IK failed - unreachable` | Reduce `--scale`; hand is outside robot's workspace |
| Jittery motion | Increase `--smoothing` (try 0.6); increase `DEADBAND_CM` to 1.5 |
| Laggy / slow response | Decrease `--smoothing` (try 0.3); check SteamVR frame rate |
| Gripper doesn't open fully | Check `GRIPPER_MAX_MM` constant matches your hardware |
| `piper_core not found` | Ensure `../teleop/piper_core.py` exists (run from `teleop_quest/`) |
