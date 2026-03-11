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

## For recording demos:
```bash 
python teleop/teleop.py --with_robot --left can0 --record ./data/tool_good --task "tool inspection good"
```

```bash
python teleop/teleop.py --with_robot --left can0 --record ./data/tool_bad --task "tool inspection bad"
```
