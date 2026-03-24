#!/usr/bin/env python3
"""Train a diffusion policy on a sub-task dataset.

Usage:
  python policy/train.py --dataset_dir ./data/pick --output_dir ./checkpoints/pick
  python policy/train.py --dataset_dir ./data/pick --output_dir ./checkpoints/pick --wandb
  python policy/train.py --dataset_dir ./data/pick --output_dir ./checkpoints/pick --resume latest
"""

import argparse
import copy
import gc
import json
import math
import os
import shutil
import subprocess
import sys
import time

# Keep CPU-side libraries on conservative ISA/threading defaults.
# Training is GPU-bound here, so this trades a little host throughput for stability.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("DNNL_MAX_CPU_ISA", "AVX2")
os.environ.setdefault("ONEDNN_MAX_CPU_ISA", "AVX2")
os.environ.setdefault("MKL_ENABLE_INSTRUCTIONS", "AVX2")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler

from config import DiffusionPolicyConfig
from normalizer import MinMaxNormalizer
from model import VisionEncoder, ConditionalUnet1D, EMAModel
from dataset import PiperDataset


def parse_args():
    p = argparse.ArgumentParser(description="Train diffusion policy")
    p.add_argument("--dataset_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--obs_horizon", type=int, default=None)
    p.add_argument("--pred_horizon", type=int, default=None)
    p.add_argument("--action_horizon", type=int, default=None)
    p.add_argument("--resume", type=str, default=None,
                   help="Checkpoint to resume from (path or 'latest')")
    p.add_argument("--epochs_per_run", type=int, default=0,
                   help="Max epochs per process before clean restart; 0 disables restarts")
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--save_every_epochs", type=int, default=None)
    p.add_argument("--save_latest_every_epochs", type=int, default=50,
                   help="How often to rewrite latest.pt (full resume state)")
    p.add_argument("--save_recovery_every_steps", type=int, default=200,
                   help="How often to save recovery.pt (weights-only crash resume)")
    p.add_argument("--sync_cuda_every_steps", type=int, default=0,
                   help="CUDA sync cadence for stability/debugging; 0 disables it")
    p.add_argument("--nice_level", type=int, default=10,
                   help="Lower CPU priority by this amount; 0 disables it")
    p.add_argument("--cpu_threads", type=int, default=None,
                   help="Max PyTorch CPU threads; default keeps desktop responsive")
    p.add_argument("--interop_threads", type=int, default=1,
                   help="PyTorch inter-op thread count")
    p.add_argument("--opencv_threads", type=int, default=1,
                   help="OpenCV thread count")
    p.add_argument("--ionice", action=argparse.BooleanOptionalAction, default=True,
                   help="Lower I/O priority on Linux when ionice is available")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def atomic_torch_save(payload, path):
    tmp_path = f"{path}.tmp"
    last_error = None
    for attempt in range(3):
        try:
            torch.save(payload, tmp_path)
            os.replace(tmp_path, path)
            return
        except Exception as exc:
            last_error = exc
            try:
                os.remove(tmp_path)
            except FileNotFoundError:
                pass
            except OSError:
                pass
            if attempt == 2:
                break
            gc.collect()
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            time.sleep(0.5 * (attempt + 1))
    raise last_error


def atomic_json_save(payload, path):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.replace(tmp_path, path)


def _module_state_dict_to_cpu(module):
    return {name: _tensor_tree_to_cpu(value) for name, value in module.state_dict().items()}


def _tensor_tree_to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _tensor_tree_to_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_tensor_tree_to_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_tensor_tree_to_cpu(v) for v in value)
    return value


def try_save_checkpoint(label, path, noise_net, vision_encoder, ema_model,
                        optimizer, lr_scheduler, state_norm, action_norm,
                        config, global_step, epoch, best_loss,
                        include_optimizer_state=True,
                        include_ema_state=True):
    try:
        save_checkpoint(
            path,
            noise_net,
            vision_encoder,
            ema_model,
            optimizer,
            lr_scheduler,
            state_norm,
            action_norm,
            config,
            global_step,
            epoch,
            best_loss,
            include_optimizer_state=include_optimizer_state,
            include_ema_state=include_ema_state,
        )
        return True
    except Exception as exc:
        print(f"  WARNING: failed to save {label}: {exc}")
        return False


def _optimizer_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device=device, non_blocking=(device.type == "cuda"))


def configure_runtime(args):
    cpu_threads = args.cpu_threads
    if cpu_threads is None:
        cpu_count = os.cpu_count() or 4
        cpu_threads = max(1, min(4, cpu_count // 2 if cpu_count > 1 else 1))
    cpu_threads = max(1, cpu_threads)

    interop_threads = max(1, args.interop_threads)
    opencv_threads = max(1, args.opencv_threads)

    if args.nice_level > 0:
        try:
            os.nice(args.nice_level)
        except OSError:
            pass

    if args.ionice and sys.platform.startswith("linux"):
        ionice_bin = shutil.which("ionice")
        if ionice_bin is not None:
            subprocess.run(
                [ionice_bin, "-c2", "-n7", "-p", str(os.getpid())],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    torch.set_num_threads(cpu_threads)
    try:
        torch.set_num_interop_threads(interop_threads)
    except RuntimeError:
        pass

    try:
        import cv2
        cv2.setNumThreads(opencv_threads)
    except Exception:
        pass

    return {
        "cpu_threads": cpu_threads,
        "interop_threads": interop_threads,
        "opencv_threads": opencv_threads,
        "nice_level": args.nice_level,
        "ionice": args.ionice,
    }


def save_checkpoint(path, noise_net, vision_encoder, ema_model,
                    optimizer, lr_scheduler, state_norm, action_norm,
                    config, global_step, epoch, best_loss,
                    include_optimizer_state=True,
                    include_ema_state=True):
    payload = {
        "noise_net": _module_state_dict_to_cpu(noise_net),
        "vision_encoder": _module_state_dict_to_cpu(vision_encoder),
        "state_normalizer": _tensor_tree_to_cpu(state_norm.state_dict()),
        "action_normalizer": _tensor_tree_to_cpu(action_norm.state_dict()),
        "config": {
            "obs_horizon": config.obs_horizon,
            "pred_horizon": config.pred_horizon,
            "action_horizon": config.action_horizon,
            "action_dim": config.action_dim,
            "obs_state_dim": config.obs_state_dim,
            "vision_feature_dim": config.vision_feature_dim,
            "down_dims": list(config.down_dims),
            "diffusion_step_embed_dim": config.diffusion_step_embed_dim,
            "kernel_size": config.kernel_size,
            "n_groups": config.n_groups,
            "cond_predict_scale": config.cond_predict_scale,
            "num_train_timesteps": config.num_train_timesteps,
            "num_inference_steps": config.num_inference_steps,
            "image_size": list(config.image_size),
        },
        "global_step": global_step,
        "epoch": epoch,
        "best_loss": best_loss,
        "ema_step": ema_model.optimization_step,
    }
    if include_ema_state:
        payload["ema_noise_net"] = _module_state_dict_to_cpu(ema_model.averaged_model)
    if include_optimizer_state:
        payload["optimizer"] = _tensor_tree_to_cpu(optimizer.state_dict())
        payload["lr_scheduler"] = _tensor_tree_to_cpu(lr_scheduler.state_dict())
    atomic_torch_save(payload, path)


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def resolve_resume_checkpoint(output_dir, resume_arg):
    if resume_arg != "latest":
        return resume_arg

    latest_path = os.path.join(output_dir, "latest.pt")
    recovery_path = os.path.join(output_dir, "recovery.pt")
    candidates = [path for path in (latest_path, recovery_path) if os.path.exists(path)]
    if not candidates:
        return latest_path
    return max(candidates, key=os.path.getmtime)


def fast_forward_lr_scheduler(lr_scheduler, step_count):
    if step_count <= 0:
        return

    if isinstance(lr_scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
        eta_min = lr_scheduler.eta_min
        t_max = lr_scheduler.T_max
        new_lrs = []
        for param_group, base_lr in zip(lr_scheduler.optimizer.param_groups, lr_scheduler.base_lrs):
            lr = eta_min + (base_lr - eta_min) * (1 + math.cos(math.pi * step_count / t_max)) / 2
            param_group["lr"] = lr
            new_lrs.append(lr)
        lr_scheduler.last_epoch = step_count
        lr_scheduler._last_lr = new_lrs
        return

    lr_scheduler.step(step_count)


def save_best_metadata(output_dir, best_loss, epoch, global_step):
    atomic_json_save({
        "best_loss": best_loss,
        "epoch": epoch,
        "global_step": global_step,
    }, os.path.join(output_dir, "best_meta.json"))


def load_best_metadata(output_dir):
    path = os.path.join(output_dir, "best_meta.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_training_status(output_dir, status, **kwargs):
    payload = {
        "status": status,
        "timestamp": time.time(),
    }
    payload.update(kwargs)
    atomic_json_save(payload, os.path.join(output_dir, "training_status.json"))


def format_eta(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


def main():
    args = parse_args()

    # Config
    cfg = DiffusionPolicyConfig()
    cfg.dataset_dir = args.dataset_dir
    cfg.output_dir = args.output_dir
    if args.epochs is not None:
        cfg.num_epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.learning_rate = args.lr
    if args.obs_horizon is not None:
        cfg.obs_horizon = args.obs_horizon
    if args.pred_horizon is not None:
        cfg.pred_horizon = args.pred_horizon
    if args.action_horizon is not None:
        cfg.action_horizon = args.action_horizon
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.save_every_epochs is not None:
        cfg.save_every_epochs = args.save_every_epochs
    cfg.use_wandb = args.wandb

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.epochs_per_run < 0:
        raise ValueError("--epochs_per_run must be >= 0")
    if args.nice_level < 0:
        raise ValueError("--nice_level must be >= 0")
    if args.interop_threads < 1:
        raise ValueError("--interop_threads must be >= 1")
    if args.opencv_threads < 1:
        raise ValueError("--opencv_threads must be >= 1")
    if args.sync_cuda_every_steps < 0:
        raise ValueError("--sync_cuda_every_steps must be >= 0")
    if args.save_latest_every_epochs < 1:
        raise ValueError("--save_latest_every_epochs must be >= 1")
    if args.save_recovery_every_steps < 1:
        raise ValueError("--save_recovery_every_steps must be >= 1")

    runtime_cfg = configure_runtime(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        # Prefer stability over maximum throughput on desktop GPUs.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    os.makedirs(cfg.output_dir, exist_ok=True)
    save_training_status(
        cfg.output_dir,
        "initializing",
        dataset_dir=cfg.dataset_dir,
        device=str(device),
        pid=os.getpid(),
    )

    # Dataset
    print(f"\n=== Loading dataset from {cfg.dataset_dir} ===")
    dataset = PiperDataset(
        cfg.dataset_dir,
        obs_horizon=cfg.obs_horizon,
        pred_horizon=cfg.pred_horizon,
        image_size=cfg.image_size,
    )
    dataloader_kwargs = {
        "batch_size": cfg.batch_size,
        "shuffle": True,
        "num_workers": cfg.num_workers,
        "pin_memory": args.pin_memory,
        "drop_last": True,
    }
    if cfg.num_workers > 0:
        dataloader_kwargs["persistent_workers"] = True
        dataloader_kwargs["prefetch_factor"] = 2
    dataloader = DataLoader(dataset, **dataloader_kwargs)

    # Normalization
    print("Computing normalization statistics...")
    state_norm = MinMaxNormalizer()
    action_norm = MinMaxNormalizer()
    state_norm.fit(dataset.get_all_states())
    action_norm.fit(dataset.get_all_actions())
    print(f"  State range: {state_norm.data_min} -> {state_norm.data_max}")
    print(f"  Action range: {action_norm.data_min} -> {action_norm.data_max}")

    # Model
    global_cond_dim = cfg.obs_horizon * (cfg.vision_feature_dim + cfg.obs_state_dim)
    print(f"\n=== Building model ===")
    print(f"  Observation: {cfg.obs_horizon} steps x "
          f"(vision {cfg.vision_feature_dim} + state {cfg.obs_state_dim}) "
          f"= {global_cond_dim}-dim conditioning")
    print(f"  Action: {cfg.pred_horizon} steps x {cfg.action_dim}-dim")

    vision_encoder = VisionEncoder(out_dim=cfg.vision_feature_dim).to(device)
    noise_net = ConditionalUnet1D(
        input_dim=cfg.action_dim,
        global_cond_dim=global_cond_dim,
        diffusion_step_embed_dim=cfg.diffusion_step_embed_dim,
        down_dims=cfg.down_dims,
        kernel_size=cfg.kernel_size,
        n_groups=cfg.n_groups,
        cond_predict_scale=cfg.cond_predict_scale,
    ).to(device)
    ema_model = EMAModel(noise_net, power=cfg.ema_power)
    ema_model.averaged_model.to(device)
    ema_model.refresh_parameter_cache(noise_net)

    n_vis = sum(p.numel() for p in vision_encoder.parameters())
    n_unet = sum(p.numel() for p in noise_net.parameters())
    print(f"  Vision encoder: {n_vis / 1e6:.1f}M params")
    print(f"  Noise U-Net: {n_unet / 1e6:.1f}M params")
    print(f"  Total: {(n_vis + n_unet) / 1e6:.1f}M params")

    # Noise scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=cfg.num_train_timesteps,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule=cfg.beta_schedule,
        clip_sample=True,
        prediction_type="epsilon",
    )

    # Optimizer
    all_params = list(vision_encoder.parameters()) + list(noise_net.parameters())
    optimizer = torch.optim.AdamW(
        all_params, lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay, betas=cfg.betas)

    total_steps = cfg.num_epochs * len(dataloader)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-6)

    start_epoch = 0
    global_step = 0
    best_loss = float("inf")

    # Resume
    if args.resume:
        ckpt_path = resolve_resume_checkpoint(cfg.output_dir, args.resume)
        if os.path.exists(ckpt_path):
            print(f"\nResuming from {ckpt_path}")
            ckpt = load_checkpoint(ckpt_path)
            is_recovery_checkpoint = os.path.basename(ckpt_path) == "recovery.pt"
            optimizer_state_restored = False
            noise_net.load_state_dict(ckpt["noise_net"])
            vision_encoder.load_state_dict(ckpt["vision_encoder"])
            ema_model.averaged_model.load_state_dict(
                ckpt.get("ema_noise_net", ckpt["noise_net"]))
            ema_model.optimization_step = ckpt.get("ema_step", ema_model.optimization_step)
            ema_model.refresh_parameter_cache(noise_net)
            if "optimizer" in ckpt and "lr_scheduler" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
                lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
                _optimizer_to_device(optimizer, device)
                optimizer_state_restored = True
            else:
                print("  Resume checkpoint has weights only; optimizer state not restored")
            state_norm.load_state_dict(ckpt["state_normalizer"])
            action_norm.load_state_dict(ckpt["action_normalizer"])
            start_epoch = ckpt["epoch"] if is_recovery_checkpoint else ckpt["epoch"] + 1
            if is_recovery_checkpoint:
                global_step = ckpt["epoch"] * len(dataloader)
                print(f"  Recovery checkpoint rewinds to epoch {start_epoch} start")
            else:
                global_step = ckpt["global_step"]
            if not optimizer_state_restored and global_step > 0:
                fast_forward_lr_scheduler(lr_scheduler, global_step)
                print(
                    f"  Fast-forwarded LR scheduler to step {global_step} "
                    f"(lr={lr_scheduler.get_last_lr()[0]:.2e})"
                )
            best_loss = ckpt["best_loss"]
            best_meta = load_best_metadata(cfg.output_dir)
            if best_meta is not None and best_meta["best_loss"] < best_loss:
                best_loss = best_meta["best_loss"]
                print(f"  Preserving lower best loss from best_meta.json: {best_loss:.6f}")
            print(f"  Resumed at epoch {start_epoch}, step {global_step}, "
                  f"best_loss={best_loss:.6f}")
            del ckpt
        else:
            print(f"WARNING: Checkpoint {ckpt_path} not found, training from scratch")

    # wandb
    if cfg.use_wandb:
        import wandb
        task_name = os.path.basename(cfg.dataset_dir)
        wandb.init(project=cfg.wandb_project, name=f"{task_name}",
                   config=vars(cfg))

    # Training loop
    steps_per_epoch = len(dataloader)
    print(f"\n=== Training ===")
    print(f"  Epochs: {cfg.num_epochs} ({start_epoch} -> {cfg.num_epochs})")
    print(f"  Steps/epoch: {steps_per_epoch}")
    print(f"  Total steps: {total_steps}")
    print(f"  Batch size: {cfg.batch_size}")
    print(f"  DataLoader workers: {cfg.num_workers}")
    print(f"  Pin memory: {'ON' if args.pin_memory else 'OFF'}")
    print(f"  CPU threads: {runtime_cfg['cpu_threads']}")
    print(f"  Inter-op threads: {runtime_cfg['interop_threads']}")
    print(f"  OpenCV threads: {runtime_cfg['opencv_threads']}")
    print(f"  Nice level: {runtime_cfg['nice_level']}")
    print(f"  Ionice: {'ON' if runtime_cfg['ionice'] else 'OFF'}")
    print(f"  latest.pt cadence: every {args.save_latest_every_epochs} epoch(s)")
    print(f"  recovery.pt cadence: every {args.save_recovery_every_steps} step(s)")
    if device.type == "cuda":
        if args.sync_cuda_every_steps > 0:
            print(f"  CUDA sync cadence: every {args.sync_cuda_every_steps} step(s)")
        else:
            print("  CUDA sync cadence: OFF")
    if args.epochs_per_run > 0:
        print(f"  Auto-restart cadence: every {args.epochs_per_run} epoch(s)")
    else:
        print("  Auto-restart cadence: OFF")
    print()
    save_training_status(
        cfg.output_dir,
        "running",
        epoch=start_epoch,
        global_step=global_step,
        best_loss=best_loss,
    )

    # Mixed precision disabled — can cause 'Illegal instruction' on some drivers
    use_amp = False
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    print(f"  Mixed precision (fp16): OFF")

    train_start = time.time()
    steps_done_this_run = 0

    for epoch in range(start_epoch, cfg.num_epochs):
        vision_encoder.train()
        noise_net.train()
        epoch_losses = []
        epoch_start = time.time()

        for batch in dataloader:
            obs_state = batch["obs_state"].to(device)   # (B, To, 7)
            obs_image = batch["obs_image"].to(device)   # (B, To, 3, H, W)
            action = batch["action"].to(device)          # (B, Tp, 7)
            B = obs_state.shape[0]
            To = cfg.obs_horizon

            # Normalize state and action
            obs_state_n = state_norm.normalize(obs_state)
            action_n = action_norm.normalize(action)

            with torch.amp.autocast("cuda", enabled=use_amp):
                # Vision encoding
                img_flat = obs_image.reshape(B * To, *obs_image.shape[2:])
                vis_feat = vision_encoder(img_flat)         # (B*To, vis_dim)
                vis_feat = vis_feat.reshape(B, To, -1)       # (B, To, vis_dim)

                # Build global conditioning: concat vision + state, flatten
                obs_feat = torch.cat([vis_feat, obs_state_n], dim=-1)
                global_cond = obs_feat.reshape(B, -1)

                # Diffusion forward process
                noise = torch.randn_like(action_n)
                timesteps = torch.randint(0, cfg.num_train_timesteps, (B,),
                                          device=device).long()
                noisy_action = noise_scheduler.add_noise(action_n, noise, timesteps)

                # Predict noise
                noise_pred = noise_net(noisy_action, timesteps, global_cond=global_cond)
                loss = F.mse_loss(noise_pred, noise)

            # Backward with scaled gradients
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(all_params, cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            ema_model.step(noise_net)

            # Optional sync for debugging CUDA hangs without stalling every step.
            if (device.type == "cuda" and args.sync_cuda_every_steps > 0
                    and global_step % args.sync_cuda_every_steps == 0):
                torch.cuda.synchronize()

            global_step += 1
            steps_done_this_run += 1
            epoch_losses.append(loss.item())

            if global_step % args.save_recovery_every_steps == 0:
                if try_save_checkpoint(
                    "recovery.pt",
                    os.path.join(cfg.output_dir, "recovery.pt"),
                    noise_net, vision_encoder, ema_model,
                    optimizer, lr_scheduler, state_norm, action_norm,
                    cfg, global_step, epoch, best_loss,
                    include_optimizer_state=False,
                    include_ema_state=False,
                ):
                    print(f"  -> Saved recovery.pt at step {global_step}")

            # Log
            if global_step % cfg.log_every_steps == 0:
                elapsed = time.time() - train_start
                steps_remaining = total_steps - global_step
                remaining = elapsed / steps_done_this_run * steps_remaining
                lr_now = lr_scheduler.get_last_lr()[0]
                print(f"  step {global_step:>6d}/{total_steps}  "
                      f"loss={loss.item():.6f}  lr={lr_now:.2e}  "
                      f"ETA={format_eta(remaining)}")
                save_training_status(
                    cfg.output_dir,
                    "running",
                    epoch=epoch + 1,
                    global_step=global_step,
                    loss=float(loss.item()),
                    lr=float(lr_now),
                    eta_seconds=float(remaining),
                    best_loss=float(best_loss),
                )
                if cfg.use_wandb:
                    import wandb
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/lr": lr_now,
                        "train/epoch": epoch,
                        "train/step": global_step,
                    }, step=global_step)

        # End of epoch
        avg_loss = np.mean(epoch_losses)
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch + 1}/{cfg.num_epochs}  "
              f"avg_loss={avg_loss:.6f}  time={epoch_time:.1f}s")
        save_training_status(
            cfg.output_dir,
            "running",
            epoch=epoch + 1,
            global_step=global_step,
            avg_loss=float(avg_loss),
            epoch_time_seconds=float(epoch_time),
            best_loss=float(best_loss),
        )

        if cfg.use_wandb:
            import wandb
            wandb.log({"train/epoch_loss": avg_loss, "train/epoch": epoch + 1},
                      step=global_step)

        # Save best
        if avg_loss < best_loss:
            candidate_best_loss = avg_loss
            if try_save_checkpoint(
                "best.pt",
                os.path.join(cfg.output_dir, "best.pt"),
                noise_net, vision_encoder, ema_model,
                optimizer, lr_scheduler, state_norm, action_norm,
                cfg, global_step, epoch, candidate_best_loss,
                include_optimizer_state=False,
                include_ema_state=True,
            ):
                best_loss = candidate_best_loss
                save_best_metadata(cfg.output_dir, best_loss, epoch, global_step)
                print(f"  -> Saved best.pt (loss={best_loss:.6f})")
            else:
                print(
                    "  WARNING: best loss improved, but checkpoint save failed; "
                    "keeping previous best.pt"
                )

        epochs_this_run = epoch - start_epoch + 1
        should_save_latest = (
            (epoch + 1) % args.save_latest_every_epochs == 0
            or (epoch + 1) == cfg.num_epochs
            or (args.epochs_per_run > 0 and epochs_this_run >= args.epochs_per_run)
        )
        if should_save_latest:
            latest_path = os.path.join(cfg.output_dir, "latest.pt")
            latest_saved = try_save_checkpoint(
                "latest.pt",
                latest_path,
                noise_net, vision_encoder, ema_model,
                optimizer, lr_scheduler, state_norm, action_norm,
                cfg, global_step, epoch, best_loss,
                include_optimizer_state=True,
                include_ema_state=True,
            )
            if latest_saved:
                print("  -> Saved latest.pt")
                save_training_status(
                    cfg.output_dir,
                    "running",
                    epoch=epoch + 1,
                    global_step=global_step,
                    best_loss=float(best_loss),
                    latest_checkpoint="latest.pt",
                )
            else:
                latest_saved = try_save_checkpoint(
                    "latest.pt (weights-only fallback)",
                    latest_path,
                    noise_net, vision_encoder, ema_model,
                    optimizer, lr_scheduler, state_norm, action_norm,
                    cfg, global_step, epoch, best_loss,
                    include_optimizer_state=False,
                    include_ema_state=True,
                )
                if latest_saved:
                    print("  -> Saved latest.pt (weights-only fallback)")
                    save_training_status(
                        cfg.output_dir,
                        "running",
                        epoch=epoch + 1,
                        global_step=global_step,
                        best_loss=float(best_loss),
                        latest_checkpoint="latest.pt",
                    )

        # Periodic memory cleanup
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Periodic checkpoint
        if (epoch + 1) % cfg.save_every_epochs == 0:
            if try_save_checkpoint(
                f"epoch_{epoch + 1:04d}.pt",
                os.path.join(cfg.output_dir, f"epoch_{epoch + 1:04d}.pt"),
                noise_net, vision_encoder, ema_model,
                optimizer, lr_scheduler, state_norm, action_norm,
                cfg, global_step, epoch, best_loss,
                include_optimizer_state=False,
                include_ema_state=True,
            ):
                print(f"  -> Saved epoch_{epoch + 1:04d}.pt")

        # Auto-restart: exit after N epochs to avoid GPU driver hang
        if (args.epochs_per_run > 0
                and epochs_this_run >= args.epochs_per_run
                and (epoch + 1) < cfg.num_epochs):
            print(f"\n  Auto-restart: completed {epochs_this_run} epochs this run.")
            print(f"  Exiting cleanly for restart (epoch {epoch + 1}/{cfg.num_epochs})...")
            save_training_status(
                cfg.output_dir,
                "restarting",
                epoch=epoch + 1,
                global_step=global_step,
                best_loss=float(best_loss),
            )
            sys.exit(42)  # special exit code for restart

    total_time = time.time() - train_start
    print(f"\n=== Training complete ===")
    print(f"  Total time: {format_eta(total_time)}")
    print(f"  Best loss: {best_loss:.6f}")
    print(f"  Checkpoints in: {cfg.output_dir}")
    save_training_status(
        cfg.output_dir,
        "completed",
        epoch=cfg.num_epochs,
        global_step=global_step,
        best_loss=float(best_loss),
        total_time_seconds=float(total_time),
    )

    if cfg.use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
