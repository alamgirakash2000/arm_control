#!/usr/bin/env python3
"""
Run finetuned OpenVLA on your Piper arm with RTX 4090.

Uses 4-bit quantization to fit 7B model in 24GB VRAM.
Accepts language instructions at runtime.

Usage:
    python openvla-ft/inference.py \
        --model_path ./checkpoints/openvla_pick \
        --can can0 --global-cam 0 --wrist-cam 2 \
        --instruction "pick the object from the table"
"""

import argparse
import math
import os
import sys
import time
import threading

import cv2
import numpy as np
import torch
from PIL import Image

# Add project root for piper_core
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "teleop"))

from piper_core import (
    PiperHangingController, HOME_POSITION,
    JOINT_LOWER, JOINT_UPPER, RAD_TO_MDEG, J6_PHYSICAL_OFFSET,
)

GRIPPER_MAX_UM = 70000
JOINT_MARGIN = math.radians(2.0)


# ── ReplayController for smooth motion ──────────────────────────────────

class ReplayController:
    """Linear interpolation at 50Hz — same as zarr_viewer."""

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


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Run OpenVLA on Piper arm")
    p.add_argument("--model_path", required=True,
                    help="Path to finetuned model directory")
    p.add_argument("--can", required=True, help="CAN port, e.g. can0")
    p.add_argument("--speed", type=int, default=70)
    p.add_argument("--global-cam", type=int, default=0)
    p.add_argument("--wrist-cam", type=int, default=2)
    p.add_argument("--instruction", default="pick the object from the table",
                    help="Language instruction for the task")
    p.add_argument("--hz", type=float, default=5,
                    help="Inference rate (OpenVLA is ~3-5 Hz)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 55)
    print("  OpenVLA INFERENCE — Piper Arm")
    print("=" * 55)

    # ── Load model with 4-bit quantization ───────────────────────
    print(f"  Loading model from {args.model_path} ...")
    print(f"  (4-bit quantization for RTX 4090)")

    from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )

    processor = AutoProcessor.from_pretrained(
        args.model_path, trust_remote_code=True)

    vla = AutoModelForVision2Seq.from_pretrained(
        args.model_path,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    vla.eval()

    # Find unnorm key from dataset statistics
    unnorm_key = None
    for f in os.listdir(args.model_path):
        if f.endswith("_statistics.json") or f == "dataset_statistics.json":
            unnorm_key = f.replace("_statistics.json", "").replace("dataset_statistics.json", "")
            break

    print(f"  Model loaded. VRAM: ~12GB")
    print(f"  Instruction: {args.instruction}")
    print(f"  Inference rate: {args.hz} Hz")

    # ── Cameras ──────────────────────────────────────────────────
    cap_g = cv2.VideoCapture(args.global_cam)
    cap_w = cv2.VideoCapture(args.wrist_cam)
    for cap in (cap_g, cap_w):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # ── Robot ────────────────────────────────────────────────────
    print(f"  Connecting to robot on {args.can} ...")
    ctrl = PiperHangingController(can_port=args.can)
    if not ctrl.connect():
        print("  FAILED"); sys.exit(1)
    ctrl.speed = args.speed
    ctrl.go_home()
    print("  Ready.\n")

    commander = ReplayController(ctrl, hz=50)
    interval = 1.0 / args.hz
    step = 0
    grip_opened = False
    grip_closed_steps = 0

    print("  Running — press Q or Ctrl+C to stop\n")

    try:
        while True:
            t0 = time.time()

            # Read cameras
            ret_g, frame_g = cap_g.read()
            ret_w, frame_w = cap_w.read()
            if not ret_g or not ret_w:
                continue

            # Convert to PIL (OpenVLA expects PIL images)
            img_g = Image.fromarray(cv2.cvtColor(frame_g, cv2.COLOR_BGR2RGB))
            img_w = Image.fromarray(cv2.cvtColor(frame_w, cv2.COLOR_BGR2RGB))

            # Run OpenVLA inference
            inputs = processor(args.instruction, img_g).to(device, dtype=torch.bfloat16)

            with torch.no_grad():
                action = vla.predict_action(
                    **inputs,
                    unnorm_key=unnorm_key,
                    do_sample=False,
                )

            # Extract joint targets + gripper
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy().flatten()

            target_q = [float(action[j]) for j in range(min(6, len(action)))]
            target_grip = float(action[6]) if len(action) > 6 else 0.0

            # Send to robot
            commander.set_target(target_q, target_grip, time.time() + interval)

            # Detect pick complete
            if target_grip > 0.5:
                grip_opened = True
                grip_closed_steps = 0
            elif grip_opened and target_grip < 0.15:
                grip_closed_steps += 1

            if grip_opened and grip_closed_steps >= int(args.hz):
                print(f"\n\n  Pick complete! Going home ...")
                break

            # Display
            if step % 2 == 0:
                q_str = " ".join(f"{math.degrees(q):+6.1f}" for q in target_q)
                print(f"\r  step {step:5d} | {q_str} | grip {target_grip:.2f}  ",
                      end="", flush=True)

            disp_g = cv2.resize(frame_g, (640, 360))
            disp_w = cv2.resize(frame_w, (640, 360))
            cv2.putText(disp_g, "global", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(disp_w, "wrist", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(disp_g, args.instruction, (10, 350),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.imshow("OpenVLA", np.hstack([disp_g, disp_w]))
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break

            elapsed = time.time() - t0
            if elapsed < interval:
                time.sleep(interval - elapsed)
            step += 1

    except KeyboardInterrupt:
        print("\n\n  Stopped.")
    finally:
        commander.stop()
        print("  Homing ...")
        ctrl.go_home()
        ctrl.go_relax()
        ctrl.shutdown()
        cap_g.release()
        cap_w.release()
        cv2.destroyAllWindows()
        print("  Done.")


if __name__ == "__main__":
    main()
