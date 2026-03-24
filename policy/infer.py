#!/usr/bin/env python3
"""Run a trained diffusion policy checkpoint on the real robot.

Usage:
  python policy/infer.py --checkpoint ./checkpoints/pick/best.pt --can can0 --camera 0
  python policy/infer.py --checkpoint ./checkpoints/pick/epoch_0200.pt --can can0 --camera 0 --max_steps 400
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
from diffusers import DDIMScheduler

# Allow imports from sibling dirs
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "teleop"))

from piper_core import (PiperHangingController, HOME_POSITION,
                         JOINT_LOWER, JOINT_UPPER, RAD_TO_MDEG, J6_PHYSICAL_OFFSET)
from model import VisionEncoder, ConditionalUnet1D
from normalizer import MinMaxNormalizer


JOINT_MARGIN = math.radians(3.0)
GRIPPER_MAX_MM = 70.0


class PolicyCommander:
    """Background thread that continuously sends smoothed joint commands at 30Hz.

    Uses per-joint EMA smoothing to match the teleop system's motion quality.
    """

    # Per-joint EMA alpha (same as teleop baseline — J4 slower to avoid twist slamming)
    JOINT_ALPHA = [0.35, 0.35, 0.35, 0.2, 0.35, 0.35]

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
                # Apply per-joint EMA smoothing
                if self._smooth_q is None:
                    self._smooth_q = list(q)
                else:
                    for j in range(6):
                        a = self.JOINT_ALPHA[j]
                        self._smooth_q[j] = a * q[j] + (1 - a) * self._smooth_q[j]
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

    # Reconstruct models
    vision_encoder = VisionEncoder(out_dim=cfg["vision_feature_dim"]).to(device)
    global_cond_dim = cfg["obs_horizon"] * (cfg["vision_feature_dim"] + cfg["obs_state_dim"])
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
    vision_encoder.load_state_dict(ckpt["vision_encoder"])
    noise_net.eval()
    vision_encoder.eval()

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
    print(f"  DDIM inference steps: {cfg['num_inference_steps']}")

    return {
        "vision_encoder": vision_encoder,
        "noise_net": noise_net,
        "scheduler": scheduler,
        "state_norm": state_norm,
        "action_norm": action_norm,
        "config": cfg,
    }


@torch.no_grad()
def predict_actions(policy, obs_state, obs_images, device="cuda"):
    """Run diffusion inference to predict action chunk.

    Args:
        policy: dict from load_policy()
        obs_state: (To, 7) numpy array
        obs_images: (To, H, W, 3) numpy array, uint8
    Returns:
        actions: (pred_horizon, 7) numpy array, unnormalized
    """
    cfg = policy["config"]
    vision_encoder = policy["vision_encoder"]
    noise_net = policy["noise_net"]
    scheduler = policy["scheduler"]
    state_norm = policy["state_norm"]
    action_norm = policy["action_norm"]

    # Preprocess state
    state_t = torch.from_numpy(obs_state).float().unsqueeze(0).to(device)  # (1, To, 7)
    state_n = state_norm.normalize(state_t)

    # Preprocess images: resize directly to model input size (no cropping)
    img_size = tuple(cfg["image_size"])
    images_processed = []
    for img in obs_images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size[1], img_size[0]),
                         interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # (3, H, W)
        images_processed.append(img)
    images = np.stack(images_processed)  # (To, 3, H, W)

    img_t = torch.from_numpy(images).float().unsqueeze(0).to(device)  # (1, To, 3, H, W)

    # Vision encoding
    To = cfg["obs_horizon"]
    img_flat = img_t.reshape(To, *img_t.shape[2:])  # (To, 3, cH, cW)
    vis_feat = vision_encoder(img_flat)               # (To, vis_dim)
    vis_feat = vis_feat.unsqueeze(0)                   # (1, To, vis_dim)

    # Global conditioning
    obs_feat = torch.cat([vis_feat, state_n], dim=-1)  # (1, To, vis+state)
    global_cond = obs_feat.reshape(1, -1)               # (1, cond_dim)

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
    parser.add_argument("--camera", type=int, default=0)
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
    control_hz = 20.0

    ctrl = None
    commander = None
    cap = None
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

        # Open camera
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print(f"Failed to open camera {args.camera}")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print(f"\n=== Running policy ===")
        print(f"  Action horizon: execute {Ta} of {Tp} predicted steps")
        print(f"  Control rate: {control_hz} Hz")
        print(f"  Max steps: {args.max_steps} ({args.max_steps / control_hz:.1f}s)")
        print(f"  Press 'q' in the display window to stop\n")

        # Pre-fill observation buffers
        obs_state_buf = deque(maxlen=To)
        obs_image_buf = deque(maxlen=To)

        for _ in range(To):
            if ctrl:
                joints = ctrl.read_joints()
            else:
                joints = list(HOME_POSITION)
            state = joints + [0.0]
            obs_state_buf.append(state)

            ret, frame = cap.read()
            if ret:
                obs_image_buf.append(frame)
            else:
                obs_image_buf.append(np.zeros((480, 640, 3), dtype=np.uint8))

        # Action queue
        action_queue = deque()
        interval = 1.0 / control_hz
        last_grip = 0.0

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

            ret, frame = cap.read()
            if ret:
                obs_image_buf.append(frame.copy())

            # If action queue empty, run inference
            if len(action_queue) == 0:
                t_infer = time.time()
                obs_s = np.array(list(obs_state_buf), dtype=np.float32)
                obs_i = np.array(list(obs_image_buf), dtype=np.uint8)
                actions = predict_actions(policy, obs_s, obs_i, device)
                infer_ms = (time.time() - t_infer) * 1000
                inference_count += 1

                # Queue first Ta actions
                for i in range(min(Ta, len(actions))):
                    action_queue.append(actions[i])

                if inference_count <= 3 or inference_count % 10 == 0:
                    print(f"  Inference #{inference_count}: {infer_ms:.1f}ms, "
                          f"queued {Ta} actions")

            # Execute next action
            action = action_queue.popleft()
            target_joints = clamp_joints(action[:6].tolist())
            gripper_val = float(np.clip(action[6], 0.0, 1.0))
            grip_mm = gripper_val * GRIPPER_MAX_MM
            last_grip = gripper_val

            if commander:
                commander.set_target(target_joints, grip_mm)

            # Display
            if ret:
                display = frame.copy()
                h, w = display.shape[:2]

                # Info overlay
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
        if cap:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
