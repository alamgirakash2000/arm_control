#!/usr/bin/env python3
"""Train a diffusion policy on a sub-task dataset.

Usage:
  python policy/train.py --dataset_dir ./data/pick --output_dir ./checkpoints/pick
  python policy/train.py --dataset_dir ./data/pick --output_dir ./checkpoints/pick --wandb
  python policy/train.py --dataset_dir ./data/pick --output_dir ./checkpoints/pick --resume latest
"""

import argparse
import copy
import json
import os
import sys
import time

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
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def save_checkpoint(path, noise_net, vision_encoder, ema_model,
                    optimizer, lr_scheduler, state_norm, action_norm,
                    config, global_step, epoch, best_loss):
    torch.save({
        "noise_net": noise_net.state_dict(),
        "vision_encoder": vision_encoder.state_dict(),
        "ema_noise_net": ema_model.averaged_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "state_normalizer": state_norm.state_dict(),
        "action_normalizer": action_norm.state_dict(),
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
    }, path)


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    return ckpt


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
    cfg.use_wandb = args.wandb

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    os.makedirs(cfg.output_dir, exist_ok=True)

    # Dataset
    print(f"\n=== Loading dataset from {cfg.dataset_dir} ===")
    dataset = PiperDataset(
        cfg.dataset_dir,
        obs_horizon=cfg.obs_horizon,
        pred_horizon=cfg.pred_horizon,
        image_size=cfg.image_size,
    )
    dataloader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=0, pin_memory=False,
        drop_last=True,
    )

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
        ckpt_path = args.resume
        if ckpt_path == "latest":
            ckpt_path = os.path.join(cfg.output_dir, "latest.pt")
        if os.path.exists(ckpt_path):
            print(f"\nResuming from {ckpt_path}")
            ckpt = load_checkpoint(ckpt_path, device)
            noise_net.load_state_dict(ckpt["noise_net"])
            vision_encoder.load_state_dict(ckpt["vision_encoder"])
            ema_model.averaged_model.load_state_dict(ckpt["ema_noise_net"])
            ema_model.optimization_step = ckpt["ema_step"]
            optimizer.load_state_dict(ckpt["optimizer"])
            lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
            state_norm.load_state_dict(ckpt["state_normalizer"])
            action_norm.load_state_dict(ckpt["action_normalizer"])
            start_epoch = ckpt["epoch"] + 1
            global_step = ckpt["global_step"]
            best_loss = ckpt["best_loss"]
            print(f"  Resumed at epoch {start_epoch}, step {global_step}, "
                  f"best_loss={best_loss:.6f}")
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
    print()

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
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(all_params, cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            ema_model.step(noise_net)

            # Force CUDA sync to prevent async op accumulation
            if device.type == "cuda":
                torch.cuda.synchronize()

            global_step += 1
            steps_done_this_run += 1
            epoch_losses.append(loss.item())

            # Log
            if global_step % cfg.log_every_steps == 0:
                elapsed = time.time() - train_start
                steps_remaining = total_steps - global_step
                remaining = elapsed / steps_done_this_run * steps_remaining
                lr_now = lr_scheduler.get_last_lr()[0]
                print(f"  step {global_step:>6d}/{total_steps}  "
                      f"loss={loss.item():.6f}  lr={lr_now:.2e}  "
                      f"ETA={format_eta(remaining)}")
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

        if cfg.use_wandb:
            import wandb
            wandb.log({"train/epoch_loss": avg_loss, "train/epoch": epoch + 1},
                      step=global_step)

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(
                os.path.join(cfg.output_dir, "best.pt"),
                noise_net, vision_encoder, ema_model,
                optimizer, lr_scheduler, state_norm, action_norm,
                cfg, global_step, epoch, best_loss)
            print(f"  -> Saved best.pt (loss={best_loss:.6f})")

        # Save latest
        save_checkpoint(
            os.path.join(cfg.output_dir, "latest.pt"),
            noise_net, vision_encoder, ema_model,
            optimizer, lr_scheduler, state_norm, action_norm,
            cfg, global_step, epoch, best_loss)

        # Periodic memory cleanup
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Periodic checkpoint
        if (epoch + 1) % cfg.save_every_epochs == 0:
            save_checkpoint(
                os.path.join(cfg.output_dir, f"epoch_{epoch + 1:04d}.pt"),
                noise_net, vision_encoder, ema_model,
                optimizer, lr_scheduler, state_norm, action_norm,
                cfg, global_step, epoch, best_loss)
            print(f"  -> Saved epoch_{epoch + 1:04d}.pt")

    total_time = time.time() - train_start
    print(f"\n=== Training complete ===")
    print(f"  Total time: {format_eta(total_time)}")
    print(f"  Best loss: {best_loss:.6f}")
    print(f"  Checkpoints in: {cfg.output_dir}")

    if cfg.use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
