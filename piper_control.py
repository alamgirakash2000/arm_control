#!/usr/bin/env python3
"""
Piper Arm Delta Controller
===========================
Send dx, dy, dz commands to move the Piper arm end-effector.
Works with RViz (fake controllers) and real robot.

Usage:
    python3 piper_control.py              # Interactive mode (simulation)
    python3 piper_control.py --real       # Interactive mode (real robot)
    python3 piper_control.py 0.05 0 0.03  # One-shot: dx=5cm, dz=3cm
"""

import sys
import os
import time
import logging
import threading
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from pymoveit2 import MoveIt2

# Suppress all ROS/MoveIt verbose logging
logging.getLogger().setLevel(logging.CRITICAL)
os.environ["RCUTILS_CONSOLE_OUTPUT_FORMAT"] = ""

# ============================================================
# PIPER ARM CONFIGURATION
# ============================================================
JOINT_NAMES = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
]
BASE_LINK = "base_link"
END_EFFECTOR_LINK = "link6"
GROUP_NAME = "arm"

HOME_POSITION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# Small offset used to force MoveIt to generate a real trajectory
PRE_HOME_POSITION = [0.05, 0.0, 0.0, 0.0, 0.0, 0.0]

DEFAULT_SPEED = 0.1


class PiperDeltaController:
    """Simple delta (dx, dy, dz) controller for the Piper arm."""

    def __init__(self):
        self.node = Node("piper_delta_controller")
        # Suppress ROS logger output
        self.node.get_logger().set_level(rclpy.logging.LoggingSeverity.FATAL)

        callback_group = ReentrantCallbackGroup()

        self.moveit2 = MoveIt2(
            node=self.node,
            joint_names=JOINT_NAMES,
            base_link_name=BASE_LINK,
            end_effector_name=END_EFFECTOR_LINK,
            group_name=GROUP_NAME,
            callback_group=callback_group,
        )

        self.moveit2.max_velocity = DEFAULT_SPEED
        self.moveit2.max_acceleration = DEFAULT_SPEED

        self.executor = MultiThreadedExecutor(num_threads=4)
        self.executor.add_node(self.node)
        self.spin_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.spin_thread.start()

        from tf2_ros import Buffer, TransformListener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node)

        time.sleep(3.0)

    def get_current_pose(self):
        """Get the current end-effector position and orientation."""
        from rclpy.duration import Duration

        try:
            transform = self.tf_buffer.lookup_transform(
                BASE_LINK, END_EFFECTOR_LINK, rclpy.time.Time(), Duration(seconds=3.0)
            )
            pos = transform.transform.translation
            rot = transform.transform.rotation
            return {
                "position": [pos.x, pos.y, pos.z],
                "orientation": [rot.x, rot.y, rot.z, rot.w],
            }
        except Exception:
            return None

    def move_delta(self, dx: float, dy: float, dz: float):
        """Move end-effector by a relative offset (meters)."""
        print(f"  Moving dx={dx:.4f} dy={dy:.4f} dz={dz:.4f} ...", end=" ", flush=True)

        pose = self.get_current_pose()
        if pose is None:
            print("FAILED (could not get current pose)")
            return False

        new_pos = [
            pose["position"][0] + dx,
            pose["position"][1] + dy,
            pose["position"][2] + dz,
        ]
        quat = pose["orientation"]

        self.moveit2.move_to_pose(
            position=new_pos,
            quat_xyzw=quat,
            cartesian=False,
        )
        self.moveit2.wait_until_executed()

        print("Done")
        return True

    def move_to(self, x: float, y: float, z: float, qx=0.0, qy=0.0, qz=0.0, qw=1.0):
        """Move end-effector to an absolute pose."""
        print(f"  Moving to x={x:.4f} y={y:.4f} z={z:.4f} ...", end=" ", flush=True)

        self.moveit2.move_to_pose(
            position=[x, y, z],
            quat_xyzw=[qx, qy, qz, qw],
            cartesian=False,
        )
        self.moveit2.wait_until_executed()
        print("Done")
        return True

    def move_joints(self, joint_positions: list):
        """Move to specific joint angles (radians)."""
        print("  Moving to joint positions ...", end=" ", flush=True)

        self.moveit2.move_to_configuration(joint_positions)
        self.moveit2.wait_until_executed()
        print("Done")
        return True

    def go_home(self):
        """Move all joints to zero (home position)."""
        print("  Going home ...", end=" ", flush=True)
        # Move to a small offset first to force a real trajectory,
        # then move to true zero. This ensures the piper driver
        # receives actual joint commands (fake controllers start at
        # zero so MoveIt won't plan a trajectory if already there).
        self.moveit2.move_to_configuration(PRE_HOME_POSITION)
        self.moveit2.wait_until_executed()
        self.moveit2.move_to_configuration(HOME_POSITION)
        self.moveit2.wait_until_executed()
        print("Done")

    def shutdown(self):
        """Clean shutdown."""
        self.executor.shutdown()
        self.node.destroy_node()


def print_pose(controller):
    """Print current end-effector pose cleanly."""
    pose = controller.get_current_pose()
    if pose is None:
        print("  Could not get current pose")
    else:
        p = pose["position"]
        o = pose["orientation"]
        print(f"  Position:    x={p[0]:.4f}  y={p[1]:.4f}  z={p[2]:.4f}")
        print(f"  Orientation: qx={o[0]:.4f} qy={o[1]:.4f} qz={o[2]:.4f} qw={o[3]:.4f}")


def interactive_mode(controller, is_real=False, robot_status="not connected"):
    """Interactive terminal interface for sending delta commands."""
    print()
    print("=" * 55)
    print("  PIPER ARM DELTA CONTROLLER")
    print("=" * 55)

    if is_real:
        if robot_status == "connected":
            print("  Robot: Connected")
        elif robot_status == "failed":
            print("  Robot: Failed to connect")
        else:
            print("  Robot: Not connected")
    else:
        print("  Mode:  Simulation")

    print()
    print("Commands:")
    print("  dx dy dz       - Move by delta (meters)")
    print("                   Example: 0.05 0 0.03")
    print("  goto x y z     - Move to absolute position")
    print("                   Example: goto 0.25 0.0 0.3")
    print("  home           - Move to home (all joints zero)")
    print("  pose           - Print current end-effector pose")
    print("  speed 0.1-1.0  - Set speed (0.1=slow, 1.0=fast)")
    print("  quit           - Exit")
    print()
    print("=" * 55)

    while True:
        try:
            user_input = input("\n> ").strip()
            if not user_input:
                continue

            parts = user_input.split()
            cmd = parts[0].lower()

            if cmd in ("quit", "q", "exit"):
                break

            elif cmd == "home":
                controller.go_home()

            elif cmd == "pose":
                print_pose(controller)

            elif cmd == "speed":
                if len(parts) != 2:
                    print("Usage: speed 0.1-1.0")
                    continue
                s = float(parts[1])
                s = max(0.05, min(1.0, s))
                controller.moveit2.max_velocity = s
                controller.moveit2.max_acceleration = s
                print(f"  Speed set to {s}")

            elif cmd == "goto":
                if len(parts) != 4:
                    print("Usage: goto x y z")
                    continue
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                controller.move_to(x, y, z)

            else:
                if len(parts) == 3:
                    dx, dy, dz = float(parts[0]), float(parts[1]), float(parts[2])
                    controller.move_delta(dx, dy, dz)
                elif len(parts) == 1:
                    dx = float(parts[0])
                    controller.move_delta(dx, 0.0, 0.0)
                else:
                    print("Enter: dx dy dz  (e.g., 0.05 0 0.03)")

        except ValueError:
            print("Invalid input. Enter numbers like: 0.05 0 0.03")
        except KeyboardInterrupt:
            print()
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    rclpy.init()

    # Parse flags
    is_real = "--real" in sys.argv
    robot_status = "not connected"
    if "--status" in sys.argv:
        idx = sys.argv.index("--status")
        if idx + 1 < len(sys.argv):
            robot_status = sys.argv[idx + 1]

    # Filter out flags to check for one-shot args
    positional = [a for a in sys.argv[1:] if not a.startswith("--")]
    # Remove the status value from positional if present
    if "--status" in sys.argv:
        idx = sys.argv.index("--status")
        if idx + 1 < len(sys.argv):
            status_val = sys.argv[idx + 1]
            if status_val in positional:
                positional.remove(status_val)

    controller = PiperDeltaController()

    # Home on startup so physical robot matches RViz
    controller.go_home()

    try:
        if len(positional) == 3:
            dx = float(positional[0])
            dy = float(positional[1])
            dz = float(positional[2])
            controller.move_delta(dx, dy, dz)
        else:
            interactive_mode(controller, is_real=is_real, robot_status=robot_status)
    finally:
        # Home before shutting down
        print("  Going home before exit ...", end=" ", flush=True)
        try:
            controller.moveit2.move_to_configuration(HOME_POSITION)
            controller.moveit2.wait_until_executed()
            print("Done")
        except Exception:
            print("Skipped")
        print("  Shutting down ...", end=" ", flush=True)
        controller.shutdown()
        try:
            rclpy.shutdown()
        except Exception:
            pass
        print("Done")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
