#!/usr/bin/env python3
"""Run a trained diffusion policy checkpoint on the real robot.

Usage:
  python policy/infer.py --checkpoint ./checkpoints/pick/best.pt --can can0 --global-cam 0 --wrist-cam 2
  python policy/infer.py --checkpoint ./checkpoints/pick/best.pt --can can0 --global-cam 0 --wrist-cam 2 --max_steps 400
"""

import argparse
import math
import os
import signal
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn
from diffusers import DDIMScheduler

# Allow imports from sibling dirs
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "teleop"))

from piper_core import (PiperHangingController, HOME_POSITION,
                         JOINT_LOWER, JOINT_UPPER, RAD_TO_MDEG, J6_PHYSICAL_OFFSET)
from model import VisionEncoder, ConditionalUnet1D
from normalizer import MinMaxNormalizer


JOINT_MARGIN = math.radians(3.0)
GRIPPER_MAX_MM = 70.0
MAX_JOINT_STEP_RAD = math.radians(2.0)   # velocity cap: max joint change per 20Hz step
GRIPPER_EMA_ALPHA = 0.3                    # gripper smoothing


class PolicyCommander:
    """Background thread that continuously sends smoothed joint commands at 30Hz.

    Uses per-joint EMA smoothing to match the teleop system's motion quality.
    """

    # Per-joint EMA alpha (lowered to match teleop-level smoothness)
    # J4 slower to avoid twist slamming, J5 also slower
    JOINT_ALPHA = [0.15, 0.15, 0.15, 0.08, 0.10, 0.15]

    def __init__(self, ctrl, hz=30):
        self._piper = ctrl.piper
        self._speed = ctrl.speed
        self._lock = threading.Lock()
        self._target_q = None
        self._smooth_q = None  # EMA-smoothed joints
        self._grip_mm = None
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def set_target(self, joints_6, grip_mm):
        with self._lock:
            self._target_q = list(joints_6)
            self._grip_mm = grip_mm

    def clear(self):
        with self._lock:
            self._target_q = None
            self._smooth_q = None
            self._grip_mm = None

    def stop(self):
        self._running = False

    def _loop(self):
        while self._running:
            with self._lock:
                q = self._target_q
                g = self._grip_mm
            if q is not None:
                # Apply per-joint EMA smoothing + velocity capping
                if self._smooth_q is None:
                    self._smooth_q = list(q)
                else:
                    for j in range(6):
                        a = self.JOINT_ALPHA[j]
                        desired = a * q[j] + (1 - a) * self._smooth_q[j]
                        # Velocity cap: limit max change per tick
                        delta = desired - self._smooth_q[j]
                        max_step = MAX_JOINT_STEP_RAD / 30  # per-tick at 30Hz
                        delta = max(-max_step, min(max_step, delta))
                        self._smooth_q[j] += delta
                q_physical = list(self._smooth_q)
                q_physical[5] += J6_PHYSICAL_OFFSET  # logical → firmware
                jcmds = [round(v * RAD_TO_MDEG) for v in q_physical]
                self._piper.MotionCtrl_2(0x01, 0x01, self._speed, 0x00)
                self._piper.JointCtrl(*jcmds)
            if g is not None:
                self._piper.GripperCtrl(round(g * 1000), 1000, 0x01, 0)
            time.sleep(1.0 / 30)


def load_policy(checkpoint_path, device="cuda"):
    """Load trained policy from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    # Camera setup
    cam_names = cfg.get("camera_names", ["global", "wrist"])
    num_cams = cfg.get("num_cameras", len(cam_names))
    vis_total_dim = num_cams * cfg["vision_feature_dim"]

    # Reconstruct models — per-camera vision encoders
    vision_encoders = nn.ModuleDict({
        cam: VisionEncoder(out_dim=cfg["vision_feature_dim"]) for cam in cam_names
    }).to(device)

    global_cond_dim = cfg["obs_horizon"] * (vis_total_dim + cfg["obs_state_dim"])
    noise_net = ConditionalUnet1D(
        input_dim=cfg["action_dim"],
        global_cond_dim=global_cond_dim,
        diffusion_step_embed_dim=cfg["diffusion_step_embed_dim"],
        down_dims=tuple(cfg["down_dims"]),
        kernel_size=cfg["kernel_size"],
        n_groups=cfg["n_groups"],
        cond_predict_scale=cfg["cond_predict_scale"],
    ).to(device)

    # Load EMA weights (better for inference)
    noise_net.load_state_dict(ckpt["ema_noise_net"])
    vision_encoders.load_state_dict(ckpt["vision_encoders"])
    noise_net.eval()
    vision_encoders.eval()

    # Load normalizers
    state_norm = MinMaxNormalizer()
    action_norm = MinMaxNormalizer()
    state_norm.load_state_dict(ckpt["state_normalizer"])
    action_norm.load_state_dict(ckpt["action_normalizer"])

    # Inference noise scheduler (DDIM for speed)
    scheduler = DDIMScheduler(
        num_train_timesteps=cfg["num_train_timesteps"],
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )
    scheduler.set_timesteps(cfg["num_inference_steps"])

    print(f"  obs_horizon={cfg['obs_horizon']}, pred_horizon={cfg['pred_horizon']}, "
          f"action_horizon={cfg['action_horizon']}")
    print(f"  cameras: {cam_names}")
    print(f"  DDIM inference steps: {cfg['num_inference_steps']}")

    return {
        "vision_encoders": vision_encoders,
        "noise_net": noise_net,
        "scheduler": scheduler,
        "state_norm": state_norm,
        "action_norm": action_norm,
        "config": cfg,
    }


@torch.no_grad()
def predict_actions(policy, obs_state, obs_images_dict, device="cuda"):
    """Run diffusion inference to predict action chunk.

    Args:
        policy: dict from load_policy()
        obs_state: (To, 7) numpy array
        obs_images_dict: {cam_name: (To, H, W, 3) numpy uint8}
    Returns:
        actions: (pred_horizon, 7) numpy array, unnormalized
    """
    cfg = policy["config"]
    vision_encoders = policy["vision_encoders"]
    noise_net = policy["noise_net"]
    scheduler = policy["scheduler"]
    state_norm = policy["state_norm"]
    action_norm = policy["action_norm"]
    cam_names = cfg.get("camera_names", ["global", "wrist"])

    # Preprocess state
    state_t = torch.from_numpy(obs_state).float().unsqueeze(0).to(device)  # (1, To, 7)
    state_n = state_norm.normalize(state_t)

    To = cfg["obs_horizon"]
    img_size = tuple(cfg["image_size"])

    # Vision encoding — per camera
    vis_feats = []
    for cam in cam_names:
        images_processed = []
        frames = obs_images_dict.get(cam, np.zeros((To, 720, 1280, 3), dtype=np.uint8))
        for img in frames:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_size[1], img_size[0]),
                             interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))  # (3, H, W)
            images_processed.append(img)
        images = np.stack(images_processed)  # (To, 3, H, W)
        img_t = torch.from_numpy(images).float().to(device)  # (To, 3, H, W)
        feat = vision_encoders[cam](img_t)     # (To, vis_dim)
        vis_feats.append(feat.unsqueeze(0))    # (1, To, vis_dim)

    vis_feat = torch.cat(vis_feats, dim=-1)  # (1, To, num_cams*vis_dim)

    # Global conditioning
    obs_feat = torch.cat([vis_feat, state_n], dim=-1)
    global_cond = obs_feat.reshape(1, -1)

    # Diffusion reverse process (DDIM)
    noisy_action = torch.randn(
        (1, cfg["pred_horizon"], cfg["action_dim"]), device=device)

    for t in scheduler.timesteps:
        noise_pred = noise_net(noisy_action, t, global_cond=global_cond)
        noisy_action = scheduler.step(noise_pred, t, noisy_action).prev_sample

    # Unnormalize
    action_pred = action_norm.unnormalize(noisy_action[0].cpu()).numpy()  # (Tp, 7)
    return action_pred


def clamp_joints(joints_6):
    """Clamp joint values to safe limits."""
    clamped = []
    for i, q in enumerate(joints_6):
        lo = JOINT_LOWER[i] + JOINT_MARGIN
        hi = JOINT_UPPER[i] - JOINT_MARGIN
        clamped.append(max(lo, min(hi, q)))
    return clamped


def main():
    parser = argparse.ArgumentParser(description="Run diffusion policy on robot")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--can", type=str, default="can0")
    parser.add_argument("--global-cam", type=int, default=0,
                        help="Global camera device ID")
    parser.add_argument("--wrist-cam", type=int, default=2,
                        help="Wrist camera device ID")
    parser.add_argument("--max_steps", type=int, default=1200,
                        help="Max control steps (at 20Hz, 1200 = 60 seconds)")
    parser.add_argument("--speed", type=int, default=50)
    parser.add_argument("--no_robot", action="store_true",
                        help="Dry run without robot (camera only)")
    args = parser.parse_args()

    # Ensure Ctrl+C works even during C-level blocking calls
    def _sigint_handler(_signum, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, _sigint_handler)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load policy
    print("Loading policy...")
    policy = load_policy(args.checkpoint, device)
    cfg = policy["config"]
    To = cfg["obs_horizon"]
    Tp = cfg["pred_horizon"]
    Ta = cfg["action_horizon"]
    cam_names = cfg.get("camera_names", ["global", "wrist"])
    control_hz = 20.0

    ctrl = None
    commander = None
    caps = {}
    step = 0
    inference_count = 0

    try:
        # Connect robot
        if not args.no_robot:
            ctrl = PiperHangingController(can_port=args.can)
            ctrl.speed = args.speed
            if not ctrl.connect():
                print("Failed to connect to robot!")
                return
            print("Homing robot...")
            ctrl.go_home()
            time.sleep(1.0)
            commander = PolicyCommander(ctrl, hz=30)

        # Open cameras
        cam_dev_map = {"global": args.global_cam, "wrist": args.wrist_cam}
        for cam in cam_names:
            dev_id = cam_dev_map.get(cam, 0)
            cap = cv2.VideoCapture(dev_id)
            if not cap.isOpened():
                print(f"WARNING: Failed to open {cam} camera (dev {dev_id})")
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            caps[cam] = cap
            print(f"  {cam} camera: dev {dev_id} (1280x720)")

        if not caps:
            print("ERROR: No cameras available!")
            return

        print(f"\n=== Running policy ===")
        print(f"  Action horizon: execute {Ta} of {Tp} predicted steps")
        print(f"  Control rate: {control_hz} Hz")
        print(f"  Max steps: {args.max_steps} ({args.max_steps / control_hz:.1f}s)")
        print(f"  Press 'q' in the display window to stop\n")

        # Pre-fill observation buffers
        obs_state_buf = deque(maxlen=To)
        obs_image_bufs = {cam: deque(maxlen=To) for cam in cam_names}

        for _ in range(To):
            if ctrl:
                joints = ctrl.read_joints()
            else:
                joints = list(HOME_POSITION)
            state = joints + [0.0]
            obs_state_buf.append(state)

            for cam in cam_names:
                if cam in caps:
                    ret, frame = caps[cam].read()
                    if ret:
                        obs_image_bufs[cam].append(frame)
                    else:
                        obs_image_bufs[cam].append(
                            np.zeros((720, 1280, 3), dtype=np.uint8))
                else:
                    obs_image_bufs[cam].append(
                        np.zeros((720, 1280, 3), dtype=np.uint8))

        # Action queue
        action_queue = deque()
        interval = 1.0 / control_hz
        last_grip = 0.0
        smooth_grip = 0.0
        prev_action_end = None

        cv2.namedWindow("Policy Inference", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Policy Inference", 960, 540)

        while step < args.max_steps:
            t_start = time.time()

            # Read new observation
            if ctrl:
                joints = ctrl.read_joints()
            else:
                joints = list(HOME_POSITION)
            state = joints + [last_grip]
            obs_state_buf.append(state)

            latest_frame = None
            for cam in cam_names:
                if cam in caps:
                    ret, frame = caps[cam].read()
                    if ret:
                        obs_image_bufs[cam].append(frame.copy())
                        if cam == "global":
                            latest_frame = frame

            # If action queue empty, run inference
            if len(action_queue) == 0:
                t_infer = time.time()
                obs_s = np.array(list(obs_state_buf), dtype=np.float32)
                obs_imgs = {
                    cam: np.array(list(obs_image_bufs[cam]), dtype=np.uint8)
                    for cam in cam_names
                }
                actions = predict_actions(policy, obs_s, obs_imgs, device)
                infer_ms = (time.time() - t_infer) * 1000
                inference_count += 1

                # Blend with previous chunk to avoid discontinuities
                n_queue = min(Ta, len(actions))
                blend_steps = min(4, n_queue)
                for i in range(n_queue):
                    a = actions[i].copy()
                    if prev_action_end is not None and i < blend_steps:
                        w = (i + 1) / (blend_steps + 1)
                        a = (1 - w) * prev_action_end + w * a
                    action_queue.append(a)
                if n_queue > 0:
                    prev_action_end = actions[n_queue - 1].copy()

                if inference_count <= 3 or inference_count % 10 == 0:
                    print(f"  Inference #{inference_count}: {infer_ms:.1f}ms, "
                          f"queued {Ta} actions")

            # Execute next action
            action = action_queue.popleft()
            target_joints = clamp_joints(action[:6].tolist())
            gripper_val = float(np.clip(action[6], 0.0, 1.0))
            smooth_grip = GRIPPER_EMA_ALPHA * gripper_val + (1 - GRIPPER_EMA_ALPHA) * smooth_grip
            grip_mm = smooth_grip * GRIPPER_MAX_MM
            last_grip = smooth_grip

            if commander:
                commander.set_target(target_joints, grip_mm)

            # Display
            if latest_frame is not None:
                display = latest_frame.copy()

                lines = [
                    f"Step: {step}/{args.max_steps}  Infer: #{inference_count}",
                    f"Joints: {' '.join(f'{q:.2f}' for q in target_joints)}",
                    f"Gripper: {grip_mm:.1f}mm ({gripper_val:.2f})",
                    f"Queue: {len(action_queue)} remaining",
                ]
                for i, txt in enumerate(lines):
                    cv2.putText(display, txt, (10, 25 + i * 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (0, 255, 0), 1, cv2.LINE_AA)

                cv2.imshow("Policy Inference", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("User quit.")
                break

            step += 1

            # Rate limit
            elapsed = time.time() - t_start
            if elapsed < interval:
                time.sleep(interval - elapsed)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        print(f"\nCompleted {step} steps, {inference_count} inference calls")
        if commander:
            commander.clear()
            commander.stop()
        if ctrl:
            print("Homing robot...")
            ctrl.go_home()
            print("Relaxing...")
            ctrl.go_relax()
            ctrl.shutdown()
        for cap in caps.values():
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
