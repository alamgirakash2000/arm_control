# Piper Arm Delta Controller

## Files
- `piper_control.py` — Main control script (interactive dx, dy, dz commands)
- `start.sh` — One-command launcher (starts MoveIt + RViz + control script)


## Usage

### Simulation (RViz with fake controllers)
```bash
conda deactivate
bash start.sh
```

### Real Robot
```bash
conda deactivate
bash start.sh --real
```

### Interactive Commands
Once running, you will see a prompt `>` where you type:

```
> 0.05 0 0          # Move 5cm in X
> 0 0.03 0          # Move 3cm in Y
> 0 0 -0.02         # Move 2cm down in Z
> 0.05 0.03 0.02    # Move diagonally
> goto 0.25 0.0 0.3 # Go to absolute position
> home               # All joints to zero
> pose               # Print current position
> speed 0.5          # Set speed (0.1 slow, 1.0 fast)
> quit               # Exit
```

### One-Shot Mode
```bash
python3 piper_control.py 0.05 0 0.03
```
Moves 5cm forward + 3cm up, then exits.




## First-Time Setup (run once)

```bash
# 1. Install pymoveit2 into your ROS 2 workspace
cd ~/ros2_ws/src
git clone https://github.com/AndrejOrsula/pymoveit2.git

# 2. Install dependencies
cd ~/ros2_ws
rosdep install -y -r -i --rosdistro humble --from-paths src

# 3. Deactivate conda first! (important)
conda deactivate

# 4. Build
source /opt/ros/humble/setup.bash
cd ~/ros2_ws
colcon build --symlink-install
source install/setup.bash
```

## Verify Your Config (important!)

Run this command and check the names match `piper_control.py`:
```bash
cat ~/ros2_ws/src/piper_ros/src/piper_with_gripper_moveit/config/*.srdf
```

You need to verify:
- Group name → `arm` (look for `<group name="arm">`)
- Joint names → `joint1` through `joint6`
- Base link → `base_link`
- End-effector → `link6` (look for `<chain base_link="..." tip_link="...">`)

If different, edit the top of `piper_control.py`.

