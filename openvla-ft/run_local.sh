#!/bin/bash
# QLoRA fine-tuning on RTX 4090 (24GB) — runs locally, no cloud needed
# Slower than A100 but free. ~25-40 hours for 50K steps.
set -e

WORKDIR=$(cd "$(dirname "$0")" && pwd)
cd "$WORKDIR/openvla"

# Find dataset
DATASET_NAME=""
for dir in "$WORKDIR"/rlds_data/*/; do
    if [ -f "${dir}dataset_info.json" ]; then
        DATASET_NAME="piper_$(basename "$dir")"
        break
    fi
done

if [ -z "$DATASET_NAME" ]; then
    echo "ERROR: No converted dataset found. Run setup.sh first."
    exit 1
fi

echo "============================================"
echo "  OpenVLA QLoRA Fine-Tuning (LOCAL)"
echo "============================================"
echo "  Dataset   : $DATASET_NAME"
echo "  GPU       : RTX 4090 (24GB) with 4-bit quantization"
echo "  Batch     : 4 x 4 grad_accum = effective 16"
echo "  Est. time : 25-40 hours"
echo "  Cost      : FREE"
echo "============================================"
echo ""

torchrun --standalone --nnodes 1 --nproc-per-node 1 \
    vla-scripts/finetune.py \
    --vla_path "openvla/openvla-7b" \
    --data_root_dir "$WORKDIR/data/pick_trimmed" \
    --dataset_name "piper_pick_trimmed" \
    --run_root_dir "$WORKDIR/output/local" \
    --batch_size 4 \
    --grad_accumulation_steps 4 \
    --learning_rate 5e-4 \
    --lora_rank 32 \
    --lora_dropout 0.0 \
    --use_lora True \
    --use_quantization True \
    --image_aug True \
    --max_steps 50000 \
    --save_steps 1000 \
    --wandb_project "openvla-piper-local" \
    --run_id_note "piper_pick_qlora"

echo ""
echo "  Training complete!"
echo "  Checkpoint: $WORKDIR/output/local/"
echo "  Next: python export_model.py --recipe local"
