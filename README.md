# Piper Arm Direct Controller

Standalone controller for the Agilex Piper 6-DOF robot arm. Talks directly to the robot via CAN bus using `piper_sdk` — no ROS, no MoveIt, no RViz needed. Starts in ~2 seconds.

## Install

```bash
pip3 install numpy python-can piper_sdk
sudo apt install can-utils
```

## Connect the Physical Robot

1. Plug in the Piper arm via USB-CAN adapter.

2. Activate the CAN interface:
```bash
sudo ip link set can0 down && sudo ip link set can0 up
```

3. Verify the connection:
```bash
candump can0    # should show CAN frames scrolling
```
Press Ctrl+C to stop.

## Run

```bash
python3 piper_direct.py
```

The robot will home (all joints to zero), then you get an interactive prompt.


## Running Dual Arms

```bash
sudo bash piper_setup.sh 
python3 piper_hanging_dual.py
```



## Keyboard Controller

```bash
python3 piper_keyboard_hanging.py
```
```
==================================================
  KEYBOARD CONTROL — HANGING  (1cm per press)
==================================================
  w/s    +X / -X  (forward / backward)
  a/d    +Y / -Y  (left / right)
  q/e    +Z / -Z  (up / down)
  r      home     (safe -> mid-range)
  t      relax    (safe -> elbow -> hang)
  o/c    gripper open / close
  [/]    speed down / up
  x      quit
==================================================

```

## Troubleshooting

**"FAILED (timeout enabling)"** — CAN interface not active. Run `can_activate.sh` again.

**"Timeout" on moves** — Target may be outside the robot's workspace. Try a smaller move.

**Permission denied on CAN** — Run `can_activate.sh` with `sudo`.

**candump shows nothing** — Check USB cable, power to the robot, and that the correct CAN port name is used.
