#!/usr/bin/env python3
"""Run VLA policy on robot with text instruction.

Usage:
    python policy/infer_vla.py \
        --checkpoint ./checkpoints/vla/best.pt \
        --can can0 \
        --instruction "pick the black object"
"""

import argparse
import math
import os
import sys
import time
import threading
from collections import deque

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from config import Config
from model_vla import VLAVisionEncoder
from model import ConditionalUnet1D, EMAModel
from normalizer import MinMaxNormalizer
from piper_core import (
    PiperHangingController, HOME_POSITION,
    JOINT_LOWER, JOINT_UPPER, RAD_TO_MDEG, J6_PHYSICAL_OFFSET,
)
from diffusers import DDIMScheduler

GRIPPER_MAX_UM = 70000
JOINT_MARGIN = math.radians(2.0)


class ReplayController:
    def __init__(self, ctrl, hz=50):
        self._piper = ctrl.piper
        self._speed = ctrl.speed
        self._lock = threading.Lock()
        self._prev_q = None
        self._next_q = None
        self._prev_grip = 0.0
        self._next_grip = 0.0
        self._t_prev = 0.0
        self._t_next = 0.0
        self._running = True
        self._hz = hz
        threading.Thread(target=self._loop, daemon=True).start()

    def set_target(self, q, grip, t_arrive):
        with self._lock:
            self._prev_q = self._next_q
            self._prev_grip = self._next_grip
            self._t_prev = self._t_next
            self._next_q = [float(v) for v in q]
            self._next_grip = float(grip)
            self._t_next = t_arrive

    def stop(self):
        self._running = False

    def _loop(self):
        while self._running:
            with self._lock:
                pq, nq = self._prev_q, self._next_q
                pg, ng = self._prev_grip, self._next_grip
                tp, tn = self._t_prev, self._t_next
            if nq is not None:
                if pq is not None and tn > tp:
                    alpha = min(1.0, max(0.0, (time.time() - tp) / (tn - tp)))
                    q = [pq[j] + alpha * (nq[j] - pq[j]) for j in range(6)]
                    g = pg + alpha * (ng - pg)
                else:
                    q, g = list(nq), ng
                try:
                    q_c = [max(JOINT_LOWER[j] + JOINT_MARGIN,
                               min(JOINT_UPPER[j] - JOINT_MARGIN, q[j]))
                           for j in range(6)]
                    q_c[5] += J6_PHYSICAL_OFFSET
                    jcmds = [round(v * RAD_TO_MDEG) for v in q_c]
                    self._piper.MotionCtrl_2(0x01, 0x01, self._speed, 0x00)
                    self._piper.JointCtrl(*jcmds)
                except Exception:
                    pass
                try:
                    self._piper.GripperCtrl(
                        round(max(0.0, min(1.0, g)) * GRIPPER_MAX_UM), 1000, 0x01, 0)
                except Exception:
                    pass
            time.sleep(1.0 / self._hz)


def preprocess_frame(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    return t


def load_policy(ckpt_path, device):
    cfg = Config()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    use_delta = ckpt.get("use_delta", False)

    text_proj_dim = 256
    vla_cond_dim = cfg.obs_horizon * (cfg.num_cameras * cfg.proj_dim + text_proj_dim + cfg.obs_state_dim)

    vla_enc = VLAVisionEncoder(
        ckpt.get("siglip_model", cfg.siglip_model),
        ckpt.get("dinov2_model", cfg.dinov2_model),
        cfg.fused_dim, cfg.proj_dim, text_proj_dim,
        freeze=ckpt.get("freeze_vision", False)).to(device)
    vla_enc.load_state_dict(ckpt["vla_encoder"])
    vla_enc.eval()

    noise_net = ConditionalUnet1D(
        input_dim=cfg.action_dim, global_cond_dim=vla_cond_dim,
        down_dims=cfg.down_dims, diffusion_step_embed_dim=cfg.diffusion_step_embed_dim,
        kernel_size=cfg.kernel_size, n_groups=cfg.n_groups,
        cond_predict_scale=cfg.cond_predict_scale,
    ).to(device)
    noise_net.load_state_dict(ckpt["ema"]["averaged_model"])
    noise_net.eval()

    state_norm = MinMaxNormalizer()
    state_norm.load_state_dict(ckpt["state_normalizer"])
    action_norm = MinMaxNormalizer()
    action_norm.load_state_dict(ckpt["action_normalizer"])

    scheduler = DDIMScheduler(
        num_train_timesteps=cfg.num_train_timesteps,
        beta_schedule=cfg.beta_schedule,
        clip_sample=False, prediction_type="epsilon",
    )
    scheduler.set_timesteps(cfg.num_inference_steps)

    return {
        "vla_enc": vla_enc, "noise_net": noise_net,
        "state_norm": state_norm, "action_norm": action_norm,
        "scheduler": scheduler, "cfg": cfg, "use_delta": use_delta,
        "vla_cond_dim": vla_cond_dim, "text_proj_dim": text_proj_dim,
    }


@torch.no_grad()
def predict_actions(obs_state_buf, img_g_buf, img_w_buf, text_feat,
                    policy, device):
    cfg = policy["cfg"]
    To = cfg.obs_horizon

    obs_state = torch.from_numpy(
        np.stack(list(obs_state_buf))).float().unsqueeze(0).to(device)
    img_g = torch.cat(list(img_g_buf), dim=0).unsqueeze(0).to(device)
    img_w = torch.cat(list(img_w_buf), dim=0).unsqueeze(0).to(device)

    obs_n = policy["state_norm"].normalize(obs_state.reshape(-1, 7)).reshape(1, To, 7)

    feat_g = policy["vla_enc"].encode_vision(img_g.reshape(-1, 3, 224, 224)).reshape(1, To, cfg.proj_dim)
    feat_w = policy["vla_enc"].encode_vision(img_w.reshape(-1, 3, 224, 224)).reshape(1, To, cfg.proj_dim)

    # Text feature expanded to match observation horizon
    text_exp = text_feat.unsqueeze(1).expand(-1, To, -1)  # (1, To, text_proj_dim)

    global_cond = torch.cat([feat_g, feat_w, text_exp, obs_n], dim=-1).reshape(1, -1)

    noisy = torch.randn((1, cfg.pred_horizon, cfg.action_dim), device=device)
    for t in policy["scheduler"].timesteps:
        pred = policy["noise_net"](noisy, t.unsqueeze(0).to(device), global_cond=global_cond)
        noisy = policy["scheduler"].step(pred, t, noisy).prev_sample

    actions = policy["action_norm"].unnormalize(noisy.reshape(-1, 7)).reshape(cfg.pred_horizon, 7)
    return actions.cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--can", required=True)
    p.add_argument("--speed", type=int, default=70)
    p.add_argument("--global-cam", type=int, default=0)
    p.add_argument("--wrist-cam", type=int, default=2)
    p.add_argument("--instruction", required=True,
                   help="Task instruction, e.g. 'pick the black object'")
    p.add_argument("--j3-offset", type=float, default=0.0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fps = 20

    print("\n" + "=" * 55)
    print("  VLA DIFFUSION POLICY — INFERENCE")
    print("=" * 55)
    print(f"  Instruction: \"{args.instruction}\"")

    policy = load_policy(args.checkpoint, device)
    cfg = policy["cfg"]
    use_delta = policy["use_delta"]
    print(f"  Action mode: {'DELTA' if use_delta else 'ABSOLUTE'}")

    # Pre-compute text embedding (once — doesn't change during execution)
    text_feat = policy["vla_enc"].encode_text([args.instruction], device)  # (1, text_proj_dim)
    print(f"  Text encoded: {text_feat.shape}")

    cap_g = cv2.VideoCapture(args.global_cam)
    cap_w = cv2.VideoCapture(args.wrist_cam)
    for cap in (cap_g, cap_w):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    ctrl = PiperHangingController(can_port=args.can)
    if not ctrl.connect():
        print("  FAILED"); sys.exit(1)
    ctrl.speed = args.speed
    ctrl.go_home()
    print("  Ready.\n")

    commander = ReplayController(ctrl, hz=50)

    obs_state_buf = deque(maxlen=cfg.obs_horizon)
    img_g_buf = deque(maxlen=cfg.obs_horizon)
    img_w_buf = deque(maxlen=cfg.obs_horizon)
    current_q = list(ctrl.read_joints()) + [0.0]

    for _ in range(cfg.obs_horizon):
        obs_state_buf.append(np.array(current_q, dtype=np.float32))
        _, fg = cap_g.read()
        _, fw = cap_w.read()
        img_g_buf.append(preprocess_frame(fg))
        img_w_buf.append(preprocess_frame(fw))

    action_queue = deque()
    interval = 1.0 / fps
    step = 0

    print(f"  Running \"{args.instruction}\" — press Q or Ctrl+C to stop\n")

    try:
        while True:
            t0 = time.time()

            joints = ctrl.read_joints()
            current_q = list(joints) + [current_q[6]]
            obs_state_buf.append(np.array(current_q, dtype=np.float32))

            ret_g, frame_g = cap_g.read()
            ret_w, frame_w = cap_w.read()
            if not ret_g or not ret_w:
                continue

            if len(action_queue) <= 2:
                img_g_buf.append(preprocess_frame(frame_g))
                img_w_buf.append(preprocess_frame(frame_w))

                raw = predict_actions(obs_state_buf, img_g_buf, img_w_buf,
                                      text_feat, policy, device)
                if use_delta:
                    abs_actions = np.zeros_like(raw)
                    pos = np.array(current_q, dtype=np.float32)
                    for i in range(cfg.pred_horizon):
                        pos = pos + raw[i]
                        abs_actions[i] = pos.copy()
                else:
                    abs_actions = raw

                for a in abs_actions[:cfg.action_horizon]:
                    action_queue.append(a)

            if len(action_queue) > 0:
                act = action_queue.popleft()
                target_q = [float(act[j]) for j in range(6)]
                target_q[2] += math.radians(args.j3_offset)
                target_grip = float(np.clip(act[6], 0.0, 1.0))
                current_q[6] = target_grip

                commander.set_target(target_q, target_grip, time.time() + interval)
                step += 1

            if step % 5 == 0:
                q_str = " ".join(f"{math.degrees(q):+6.1f}" for q in target_q)
                print(f"\r  step {step:5d} | {q_str} | "
                      f"grip {target_grip:.2f} | q={len(action_queue)}  ",
                      end="", flush=True)

            disp_g = cv2.resize(frame_g, (640, 360))
            disp_w = cv2.resize(frame_w, (640, 360))
            cv2.putText(disp_g, f"Task: {args.instruction}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(disp_w, "wrist", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("VLA Policy", np.hstack([disp_g, disp_w]))
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break

            elapsed = time.time() - t0
            if elapsed < interval:
                time.sleep(interval - elapsed)

    except KeyboardInterrupt:
        print("\n\n  Stopped by user.")
        commander.stop()
        ctrl.go_home()
        ctrl.go_relax()
        ctrl.shutdown()
        cap_g.release()
        cap_w.release()
        cv2.destroyAllWindows()
        return

    # Normal stop: lift + home
    commander.stop()
    print("\n  Lifting ...")
    cur = ctrl.read_joints()
    safe = list(cur)
    safe[2] = math.radians(-60)
    safe_phys = list(safe)
    safe_phys[5] += J6_PHYSICAL_OFFSET
    jcmds = [round(v * RAD_TO_MDEG) for v in safe_phys]
    ctrl.piper.MotionCtrl_2(0x01, 0x01, ctrl.speed, 0x00)
    ctrl.piper.JointCtrl(*jcmds)
    time.sleep(2.0)
    ctrl.go_home()
    cap_g.release()
    cap_w.release()
    cv2.destroyAllWindows()
    print("  Ready for next task.")


if __name__ == "__main__":
    main()
