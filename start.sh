#!/bin/bash
# ============================================================
# Piper Arm Launcher
# ============================================================
# Usage:
#   bash start.sh              # RViz simulation (fake controllers)
#   bash start.sh --real       # Real robot (physical arm + MoveIt)
#
# Ctrl+C to stop everything cleanly.
# ============================================================

set -e

# ---- Configuration ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTROL_SCRIPT="$SCRIPT_DIR/piper_control.py"
CAN_PORT="can0"
USE_REAL_ROBOT=false

# Parse arguments
if [[ "$1" == "--real" ]]; then
    USE_REAL_ROBOT=true
fi

# ---- Source ROS 2 ----
echo "  Connecting to ROS 2 ..."
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
echo "  Connecting to ROS 2 ... Done"

# Track background PIDs for cleanup
PIDS=()

cleanup() {
    echo ""
    echo "Shutting down..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -INT "$pid" 2>/dev/null
            sleep 0.5
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    echo "Done."
    exit 0
}

trap cleanup SIGINT SIGTERM

# ---- Step 1: Start real robot driver (only in --real mode) ----
if [ "$USE_REAL_ROBOT" = true ]; then
    echo "  Connecting to physical robot on $CAN_PORT ..."
    ros2 launch piper start_single_piper.launch.py can_port:=$CAN_PORT auto_enable:=true > /dev/null 2>&1 &
    PIDS+=($!)
    sleep 5

    # Check if driver is still running
    if kill -0 "${PIDS[-1]}" 2>/dev/null; then
        ROBOT_STATUS="connected"
        echo "  Connecting to physical robot on $CAN_PORT ... Done"
    else
        ROBOT_STATUS="failed"
        echo "  Connecting to physical robot on $CAN_PORT ... Failed"
    fi
fi

# ---- Step 2: Start MoveIt 2 + RViz (suppress output) ----
echo "  Opening RViz + MoveIt ..."
ros2 launch piper_with_gripper_moveit demo.launch.py > /dev/null 2>&1 &
PIDS+=($!)
sleep 15
echo "  Opening RViz + MoveIt ... Done"

# ---- Step 3: Run control script ----
if [ "$USE_REAL_ROBOT" = true ]; then
    python3 "$CONTROL_SCRIPT" --real --status "$ROBOT_STATUS"
else
    python3 "$CONTROL_SCRIPT"
fi

# Cleanup when control script exits
cleanup
