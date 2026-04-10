#!/usr/bin/env python3
"""Export finetuned OpenVLA model by merging LoRA weights.

Produces a standalone HuggingFace model directory that can be
downloaded and run locally on your RTX 4090.

Usage:
    python export_model.py --recipe lora
    python export_model.py --recipe oft
"""

import argparse
import glob
import os
import shutil
import sys

import torch


def find_latest_checkpoint(run_dir):
    """Find the latest checkpoint directory."""
    ckpt_dirs = sorted(glob.glob(os.path.join(run_dir, "**", "checkpoint-*"), recursive=True))
    if not ckpt_dirs:
        # Try looking for adapter files directly
        adapter_dirs = sorted(glob.glob(os.path.join(run_dir, "**", "adapter_config.json"),
                                        recursive=True))
        if adapter_dirs:
            return os.path.dirname(adapter_dirs[-1])
        return None
    return ckpt_dirs[-1]


def merge_lora(run_dir, output_dir):
    """Merge LoRA adapter weights into base model."""
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from peft import PeftModel

    ckpt_dir = find_latest_checkpoint(run_dir)
    if ckpt_dir is None:
        print(f"  ERROR: No checkpoint found in {run_dir}")
        sys.exit(1)

    print(f"  Checkpoint: {ckpt_dir}")

    # Load base model
    print("  Loading base OpenVLA-7B model...")
    base_model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # Load processor
    processor = AutoProcessor.from_pretrained(
        "openvla/openvla-7b",
        trust_remote_code=True,
    )

    # Load and merge LoRA
    print("  Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, ckpt_dir)

    print("  Merging weights...")
    model = model.merge_and_unload()

    # Save merged model
    print(f"  Saving merged model to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    # Copy dataset statistics
    workdir = os.path.dirname(os.path.abspath(__file__))
    for stats_file in glob.glob(os.path.join(workdir, "rlds_data", "*_statistics.json")):
        shutil.copy(stats_file, output_dir)
    for stats_file in glob.glob(os.path.join(workdir, "rlds_data", "*", "dataset_statistics.json")):
        dst = os.path.join(output_dir, "dataset_statistics.json")
        shutil.copy(stats_file, dst)

    print(f"\n  Export complete!")
    print(f"  Model saved to: {output_dir}")
    print(f"  Size: {sum(f.stat().st_size for f in os.scandir(output_dir) if f.is_file()) / 1e9:.1f} GB")
    print(f"\n  Download to your PC:")
    print(f"    scp -r root@<POD_IP>:{output_dir} ./checkpoints/openvla_pick/")


def main():
    parser = argparse.ArgumentParser(description="Export finetuned OpenVLA model")
    parser.add_argument("--recipe", choices=["lora", "oft", "local"], default="local",
                        help="Which training recipe was used")
    args = parser.parse_args()

    workdir = os.path.dirname(os.path.abspath(__file__))
    run_dir = os.path.join(workdir, "output", args.recipe)
    output_dir = os.path.join(workdir, "output", "merged_model")

    if not os.path.isdir(run_dir):
        print(f"  ERROR: Training output not found at {run_dir}")
        print(f"  Run training first: bash run_{args.recipe}.sh")
        sys.exit(1)

    print(f"\n  Exporting {args.recipe.upper()} finetuned model...\n")
    merge_lora(run_dir, output_dir)


if __name__ == "__main__":
    main()
