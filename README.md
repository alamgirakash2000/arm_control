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
sudo bash ~/.local/lib/python3.10/site-packages/piper_sdk/can_activate.sh can0
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

### Options

```bash
python3 piper_direct.py --can can1    # use a different CAN port
python3 piper_direct.py 5 0 3        # one-shot: move 5cm X, 3cm Z, then exit
```

## Commands

All positions are in **centimeters**.

```
> 5 0 0            # move 5cm in +X
> 0 10 0           # move 10cm in +Y
> 0 0 -5           # move 5cm down in Z
> 5 3 2            # move diagonally
> -10              # shorthand: move -10cm in X (Y=0, Z=0)
> goto 25 0 30     # go to absolute position (cm)
> goto 25 0 30 0 85 0   # with orientation (roll pitch yaw in degrees)
> home             # all joints to zero
> pose             # print current end-effector position
> joints           # print current joint angles
> speed 50         # set speed 1-100%
> gripper open     # open gripper
> gripper close    # close gripper
> gripper 35       # set gripper opening in mm (0-70)
> quit             # exit (homes before shutdown)
```

## Troubleshooting

**"FAILED (timeout enabling)"** — CAN interface not active. Run `can_activate.sh` again.

**"Timeout" on moves** — Target may be outside the robot's workspace. Try a smaller move.

**Permission denied on CAN** — Run `can_activate.sh` with `sudo`.

**candump shows nothing** — Check USB cable, power to the robot, and that the correct CAN port name is used.
