# Piper Arm Direct Controller

Standalone controller for the Agilex Piper 6-DOF robot arm. Talks directly to the robot via CAN bus using `piper_sdk` — no ROS, no MoveIt, no RViz needed. Starts in ~2 seconds.

## Install

```bash
pip3 install numpy python-can piper_sdk
sudo apt install can-utils
```

## Connect the Physical Robot
```bash
sudo bash setup.sh 
```


## Run

```bash
python3 single_arm.py
```

The robot will home (all joints to zero), then you get an interactive prompt.


## Running Dual Arms

```bash
python3 dual_arm.py
```



## Keyboard Controller

```bash
python single_keyboard.py
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
