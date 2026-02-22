# Piper Dual-Arm Teleop

Real-time bilateral teleoperation of two hanging [AgileX Piper](https://github.com/agilexrobotics/piper_sdk) arms using an Apple Vision Pro headset.

---

## Hardware Requirements

| Component | Notes |
|---|---|
| 2× AgileX Piper robot arms | Mounted upside-down (hanging) |
| 2× USB-to-CAN adapters | One per arm |
| Apple Vision Pro | Running the **Tracking Streamer** app |
| Linux machine | Connected to both CAN adapters and same Wi-Fi as Vision Pro |

---

## Software Requirements

```bash
pip install avp_stream opencv-python piper_sdk
```

---

## Files

| File | Purpose |
|---|---|
| `teleop_receiver.py` | **Main teleop script** — Vision Pro → dual Piper arms |
| `dual_piper.py` | Manual keyboard controller for both arms (useful for testing) |
| `piper_core.py` | DH parameters, FK/IK solver, `PiperHangingController` class |
| `start_teleop.sh` | One-time CAN setup + arm identification helper |

---

## Quick Start

### 1. CAN setup (once per session, or once ever)

```bash
sudo bash start_teleop.sh
```

This brings up the CAN interfaces, identifies which arm is LEFT/RIGHT by briefly flashing the gripper, and saves the result to `~/.piper_arms.conf` and `/tmp/piper_arms.env`.

To re-identify arms (e.g. after swapping cables):

```bash
sudo bash start_teleop.sh --reset
```

### 2. Simulation mode — safe, no robot motion

Test the Vision Pro connection and IK validation without touching the arms:

```bash
python3 teleop_receiver.py
python3 teleop_receiver.py --ip 192.168.1.100   # if Vision Pro IP differs
python3 teleop_receiver.py --scale 0.5          # half-range motion mapping
```

The terminal and the Tracking Streamer overlay both show live deltas and IK status for each hand. Nothing moves.

### 3. Robot mode — actual arm teleoperation

```bash
python3 teleop_receiver.py --with_robot
python3 teleop_receiver.py --with_robot --left can0 --right can1   # explicit ports
python3 teleop_receiver.py --with_robot --speed 15 --scale 0.6     # conservative start
```

### 4. Manual keyboard control (no headset needed)

```bash
python3 dual_piper.py
python3 dual_piper.py --left can0 --right can1
```

---

## All CLI Options — `teleop_receiver.py`

| Flag | Default | Description |
|---|---|---|
| `--with_robot` | off | Enable real CAN commands — without this, simulation only |
| `--left IFACE` | auto | CAN interface for left arm (e.g. `can0`) |
| `--right IFACE` | auto | CAN interface for right arm (e.g. `can1`) |
| `--ip ADDR` | `10.0.0.143` | Vision Pro IP address |
| `--scale N` | `1.0` | Motion scale: `0.5` = half, `2.0` = double |
| `--speed N` | `20` | Robot speed 1–100 % |

---

## Calibration Procedure

### How the mapping works

On startup, after a 10-second countdown, the script snapshots both wrist poses as the **hand home**. Simultaneously, `--with_robot` homes the arms to their **robot home** (`HOME_POSITION` in `piper_core.py`).

Every frame thereafter:

```
robot_target = robot_home + remap(hand_current - hand_home) * scale
```

This is an **absolute position** mapping — the robot always tracks the total displacement from home, not incremental deltas. This prevents drift.

### Step 1 — Find your axis mapping

The Vision Pro wrist matrices use: **X = right, Y = up, Z = backward** (into the screen from user's perspective).

Run simulation mode and move one hand axis at a time:

```
Hand moves UP        → observe which delta changes → that VP axis maps to robot Z
Hand moves FORWARD   → observe which delta changes → that VP axis maps to robot X
Hand moves SIDEWAYS  → observe which delta changes → that VP axis maps to robot Y
```

Edit `AXIS_MAP_L` and `AXIS_MAP_R` at the top of `teleop_receiver.py`:

```python
# Format: [(VP_axis_index, sign), ...]  for [robot_x, robot_y, robot_z]
AXIS_MAP_L = [(1, 1), (0, 1), (2, -1)]   # default — tune to your setup
AXIS_MAP_R = [(1, 1), (0, -1), (2, -1)]  # right arm: lateral axis mirrored
```

### Step 2 — Verify IK in simulation

Before enabling the robot, run simulation mode and move your hands through the full expected workspace. The overlay and terminal should show **IK OK** (green). If you see `IK failed` or `joint limits exceeded`, reduce `--scale` or adjust the axis mapping.

### Step 3 — First robot run

```bash
python3 teleop_receiver.py --with_robot --speed 10 --scale 0.3
```

Start with very low speed and scale. Confirm each arm moves in the correct direction, then increase gradually.

---

## Tuning Parameters

These live at the top of `teleop_receiver.py`:

| Constant | Default | Effect |
|---|---|---|
| `DEADBAND_CM` | `1.0` | Position changes smaller than this (cm) are ignored — reduces tremor |
| `DEADBAND_DEG` | `2.0` | Rotation changes smaller than this (deg) are ignored |
| `PINCH_MAX_M` | `0.08` | Thumb-index pinch distance that maps to 100% gripper open |
| `GRIPPER_DEADBAND` | `2.0` | Gripper % change threshold before sending CAN command |
| `POLL_HZ` | `90` | Vision Pro data polling rate |
| `DISPLAY_HZ` | `30` | Terminal / overlay refresh rate |
| `ArmCommander.CMD_TIMEOUT_S` | `0.05` | Max wait per CAN command → ~20 Hz robot command rate |

**For shaky hands:** increase `DEADBAND_CM` to `1.5`–`2.0`.
**For sluggish response:** lower `DEADBAND_CM` to `0.5`.
**For overshooting:** lower `--scale` or `--speed`.

---

## Architecture

```
Vision Pro (Tracking Streamer app)
         │  avp_stream  (Wi-Fi)
         ▼
teleop_receiver.py  ─── poll at 90 Hz ───►  Left/Right wrist 4×4 SE(3)
         │
         ├─ remap()          VP axes → robot EE axes  (AXIS_MAP_L / R)
         ├─ deadband()       tremor suppression
         ├─ rotation_delta() R_cur @ R_home.T  → rotation error
         ├─ validate()       IK check:  home + delta  reachable?
         │                   ► ik_solve() → joints_in_limits() → FK round-trip
         │
         ├─ [simulation]  display only — terminal + overlay
         │
         └─ [--with_robot]
              │
              ├─ ArmCommander("left")   ← background thread, non-blocking
              │       └─ send_cartesian(x, y, z, r, p, yaw, timeout=0.05s)
              │
              └─ ArmCommander("right")  ← background thread, non-blocking
                      └─ send_cartesian(x, y, z, r, p, yaw, timeout=0.05s)
```

**Why `ArmCommander`?**
`send_cartesian` blocks for up to `timeout` seconds while the robot moves and confirms. Without background threads, the 90 Hz Vision Pro loop would stall to ~3 Hz (1/0.3 s). `ArmCommander` keeps the loop running at full rate by accepting the latest target and executing it in parallel for both arms simultaneously.

**Why absolute position, not delta?**
The Vision Pro gives absolute wrist poses. Mapping `(hand_pos - hand_home)` to `(robot_target - robot_home)` means the robot always mirrors the total hand displacement — there is no accumulating drift. A pure delta approach would drift with every small tracking noise spike.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `avp_stream not installed` | `pip install avp_stream` |
| Vision Pro connection fails | Check IP in Settings → Wi-Fi on headset; must be on same network |
| `Could not determine CAN ports` | Run `sudo bash start_teleop.sh` first, or pass `--left can0 --right can1` |
| Arm moves wrong axis | Calibrate `AXIS_MAP_L` / `AXIS_MAP_R` (see Calibration above) |
| Arm moves opposite direction | Negate the sign in the relevant `AXIS_MAP` entry |
| `IK failed - unreachable` | Reduce `--scale`; hand is outside robot's reachable workspace |
| `joint limits exceeded` | Reduce `--scale`; robot home might need adjustment |
| Arms jitter | Increase `DEADBAND_CM` / `DEADBAND_DEG` |
| Arms lag behind hands | Increase `--speed`; decrease `DEADBAND_CM`; confirm CAN bitrate is 1 Mbps |
| Only one arm moves | Check both CAN connections; confirm LEFT/RIGHT identification was correct |
| Overlay not visible | `pip install opencv-python`; check Tracking Streamer is open on headset |
