#!/usr/bin/env python3
"""Train VLA: SigLIP(vision+text) + DINOv2 + Diffusion Policy.

Same as train.py but with text conditioning from SigLIP text encoder.

Usage:
    python policy/train_vla.py \
        --dataset_dir ./data/vla_dataset \
        --output_dir ./checkpoints/vla \
        --epochs 500

    # With frozen vision (fast)
    python policy/train_vla.py \
        --dataset_dir ./data/vla_dataset \
        --output_dir ./checkpoints/vla_frozen \
        --freeze_vision --epochs 3000
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from dataset_vla import VLADataset
from model_vla import VLAVisionEncoder
from model import ConditionalUnet1D, EMAModel
from normalizer import MinMaxNormalizer


def save_checkpoint(path, epoch, step, best_loss, vla_enc, noise_net,
                    ema, optimizer, lr_sched, state_norm, action_norm, cfg):
    tmp = path + ".tmp"
    torch.save({
        "epoch": epoch,
        "global_step": step,
        "best_loss": best_loss,
        "vla_encoder": vla_enc.state_dict(),
        "noise_net": noise_net.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_sched.state_dict(),
        "state_normalizer": state_norm.state_dict(),
        "action_normalizer": action_norm.state_dict(),
        "use_delta": cfg.use_delta,
        "siglip_model": cfg.siglip_model,
        "dinov2_model": cfg.dinov2_model,
        "freeze_vision": cfg.freeze_vision,
        "is_vla": True,
    }, tmp)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser(description="Train VLA Diffusion Policy")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--freeze_vision", action="store_true")
    parser.add_argument("--delta", action="store_true")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cfg = Config()
    if args.freeze_vision:
        cfg.freeze_vision = True
        if not args.batch_size:
            cfg.batch_size = 64
    if args.delta:
        cfg.use_delta = True
    if args.batch_size:
        cfg.batch_size = args.batch_size
    cfg.num_epochs = args.epochs
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # VLA has text conditioning: +256 to the conditioning dim
    text_proj_dim = 256
    vla_cond_dim = cfg.obs_horizon * (cfg.num_cameras * cfg.proj_dim + text_proj_dim + cfg.obs_state_dim)

    print()
    print("=" * 60)
    print("  VLA: SigLIP + DINOv2 + TEXT DIFFUSION POLICY")
    print("=" * 60)
    print(f"  Device      : {device}")
    print(f"  Dataset     : {args.dataset_dir}")
    print(f"  Epochs      : {cfg.num_epochs}")
    print(f"  Batch size  : {cfg.batch_size}")
    print(f"  Action mode : {'delta' if cfg.use_delta else 'absolute'}")
    print(f"  Vision      : SigLIP + DINOv2 ({'frozen' if cfg.freeze_vision else 'finetuned'})")
    print(f"  Text        : SigLIP text encoder (frozen)")
    print(f"  Cond dim    : {vla_cond_dim}")
    print()

    # ── Dataset ──────────────────────────────────────────────────
    zarr_path = os.path.join(args.dataset_dir, "dataset.zarr")
    labels_path = os.path.join(args.dataset_dir, "meta", "task_labels.json")
    if not os.path.exists(labels_path):
        print(f"  ERROR: {labels_path} not found")
        print(f"  Run partition_vla.py --export first")
        sys.exit(1)

    dataset = VLADataset(zarr_path, labels_path,
                         cfg.obs_horizon, cfg.pred_horizon, cfg.use_delta)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True,
                            num_workers=4, pin_memory=True, persistent_workers=True)
    print(f"  Batches/ep  : {len(dataloader)}")

    # ── Normalizers ──────────────────────────────────────────────
    import zarr
    root = zarr.open_group(zarr_path, mode="r")
    all_states = np.array(root["data/state"][:], dtype=np.float32)
    all_actions = np.array(root["data/action"][:], dtype=np.float32)

    state_norm = MinMaxNormalizer()
    state_norm.fit(all_states)
    action_norm = MinMaxNormalizer()
    if cfg.use_delta:
        all_deltas = all_actions - all_states
        action_norm.fit(all_deltas)
        del all_deltas
    else:
        action_norm.fit(all_actions)
    del all_states, all_actions

    # ── Model ────────────────────────────────────────────────────
    vla_enc = VLAVisionEncoder(
        cfg.siglip_model, cfg.dinov2_model,
        cfg.fused_dim, cfg.proj_dim, text_proj_dim,
        freeze=cfg.freeze_vision).to(device)

    noise_net = ConditionalUnet1D(
        input_dim=cfg.action_dim,
        global_cond_dim=vla_cond_dim,
        down_dims=cfg.down_dims,
        diffusion_step_embed_dim=cfg.diffusion_step_embed_dim,
        kernel_size=cfg.kernel_size,
        n_groups=cfg.n_groups,
        cond_predict_scale=cfg.cond_predict_scale,
    ).to(device)
    ema = EMAModel(noise_net, power=cfg.ema_power)

    # Parameter groups
    proj_params = list(vla_enc.vision_projection.parameters()) + \
                  list(vla_enc.text_projection.parameters())
    unet_params = list(noise_net.parameters())

    if cfg.freeze_vision:
        trainable_groups = [
            {"params": proj_params, "lr": cfg.learning_rate},
            {"params": unet_params, "lr": cfg.learning_rate},
        ]
        n_trainable = sum(p.numel() for g in trainable_groups for p in g["params"])
        print(f"  Trainable   : {n_trainable:,} (vision frozen)")
    else:
        vision_params = list(vla_enc.siglip_vision.parameters()) + \
                        list(vla_enc.dinov2.parameters())
        trainable_groups = [
            {"params": vision_params, "lr": cfg.vision_lr},
            {"params": proj_params, "lr": cfg.learning_rate},
            {"params": unet_params, "lr": cfg.learning_rate},
        ]
        n_trainable = sum(p.numel() for g in trainable_groups for p in g["params"])
        print(f"  Trainable   : {n_trainable:,} (vision finetuned)")

    optimizer = torch.optim.AdamW(trainable_groups,
                                  weight_decay=cfg.weight_decay, betas=cfg.betas)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.num_epochs)
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=cfg.num_train_timesteps,
        beta_schedule=cfg.beta_schedule,
        clip_sample=False, prediction_type="epsilon",
    )

    # ── Resume ───────────────────────────────────────────────────
    start_epoch = 0
    best_loss = float("inf")
    global_step = 0
    os.makedirs(args.output_dir, exist_ok=True)

    if args.resume:
        ckpt_path = args.resume
        if args.resume == "latest":
            ckpt_path = os.path.join(args.output_dir, "latest.pt")
        if os.path.exists(ckpt_path):
            print(f"\n  Resuming from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            vla_enc.load_state_dict(ckpt["vla_encoder"])
            noise_net.load_state_dict(ckpt["noise_net"])
            ema.load_state_dict(ckpt["ema"])
            optimizer.load_state_dict(ckpt["optimizer"])
            lr_sched.load_state_dict(ckpt["lr_scheduler"])
            state_norm.load_state_dict(ckpt["state_normalizer"])
            action_norm.load_state_dict(ckpt["action_normalizer"])
            start_epoch = ckpt["epoch"] + 1
            best_loss = ckpt.get("best_loss", float("inf"))
            global_step = ckpt.get("global_step", 0)
            print(f"  Epoch {start_epoch}, step {global_step}, best {best_loss:.6f}")

    # ── Training ─────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Training — {cfg.num_epochs - start_epoch} epochs remaining")
    print(f"{'─'*60}\n")

    for epoch in range(start_epoch, cfg.num_epochs):
        if not cfg.freeze_vision:
            vla_enc.train()
        noise_net.train()
        epoch_loss = 0.0
        n_batches = 0
        t_epoch = time.time()

        for batch in dataloader:
            obs_state = batch["obs_state"].to(device)
            img_g = batch["obs_img_global"].to(device)
            img_w = batch["obs_img_wrist"].to(device)
            action = batch["action"].to(device)
            texts = batch["text"]  # list of strings

            B, To = obs_state.shape[:2]
            Tp = cfg.pred_horizon

            with torch.autocast("cuda", dtype=torch.bfloat16):
                obs_n = state_norm.normalize(obs_state.reshape(-1, 7)).reshape(B, To, 7)
                act_n = action_norm.normalize(action.reshape(-1, 7)).reshape(B, Tp, 7)

                # Vision features
                feat_g = vla_enc.encode_vision(img_g.reshape(-1, 3, 224, 224))
                feat_g = feat_g.reshape(B, To, cfg.proj_dim)
                feat_w = vla_enc.encode_vision(img_w.reshape(-1, 3, 224, 224))
                feat_w = feat_w.reshape(B, To, cfg.proj_dim)

                # Text features (same for all timesteps in observation window)
                text_feat = vla_enc.encode_text(texts, device)  # (B, text_proj_dim)
                text_feat = text_feat.unsqueeze(1).expand(-1, To, -1)  # (B, To, text_proj_dim)

                # Global conditioning: [vision_g, vision_w, text, state] × To
                obs_feat = torch.cat([feat_g, feat_w, text_feat, obs_n], dim=-1)
                global_cond = obs_feat.reshape(B, -1)

                # Diffusion
                noise = torch.randn_like(act_n)
                timesteps = torch.randint(0, cfg.num_train_timesteps, (B,), device=device).long()
                noisy = noise_scheduler.add_noise(act_n, noise, timesteps)
                pred = noise_net(noisy, timesteps, global_cond=global_cond)
                loss = F.mse_loss(pred, noise)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for g in trainable_groups for p in g["params"]], cfg.grad_clip_norm)
            optimizer.step()
            ema.update(noise_net)

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            if global_step % cfg.log_every_steps == 0:
                print(f"\r  ep {epoch:4d} | step {global_step:6d} | "
                      f"loss {loss.item():.6f} | avg {epoch_loss/n_batches:.6f} | "
                      f"lr {optimizer.param_groups[-1]['lr']:.2e}",
                      end="", flush=True)

        lr_sched.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t_epoch
        remaining = (cfg.num_epochs - epoch - 1) * elapsed
        eta_m, eta_s = divmod(int(remaining), 60)
        eta_h, eta_m = divmod(eta_m, 60)

        marker = " *" if avg_loss < best_loss else ""
        print(f"\r  ep {epoch:4d} | loss {avg_loss:.6f}{marker} | "
              f"lr {optimizer.param_groups[-1]['lr']:.2e} | "
              f"{elapsed:.1f}s | ETA {eta_h}h{eta_m:02d}m               ")

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(os.path.join(args.output_dir, "best.pt"),
                            epoch, global_step, best_loss, vla_enc, noise_net,
                            ema, optimizer, lr_sched, state_norm, action_norm, cfg)

        if (epoch + 1) % cfg.save_every_epochs == 0 or epoch == cfg.num_epochs - 1:
            save_checkpoint(os.path.join(args.output_dir, "latest.pt"),
                            epoch, global_step, best_loss, vla_enc, noise_net,
                            ema, optimizer, lr_sched, state_norm, action_norm, cfg)
            print(f"    checkpoint saved (epoch {epoch})")

    print(f"\n  Training complete. Best loss: {best_loss:.6f}")


if __name__ == "__main__":
    main()
