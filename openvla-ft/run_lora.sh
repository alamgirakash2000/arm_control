#!/bin/bash
# Cloud LoRA fine-tuning: 1x A100 80GB, ~10-15 hours, ~$15-22
set -e

WORKDIR=$(cd "$(dirname "$0")" && pwd)
cd "$WORKDIR/openvla"

echo "============================================"
echo "  OpenVLA LoRA Fine-Tuning (CLOUD)"
echo "============================================"
echo "  GPU: 1x A100 80GB"
echo "  Estimated time: 10-15 hours"
echo "============================================"
echo ""

torchrun --standalone --nnodes 1 --nproc-per-node 1 \
    vla-scripts/finetune.py \
    --vla_path "openvla/openvla-7b" \
    --data_root_dir "$WORKDIR/data/pick_trimmed" \
    --dataset_name "piper_pick_trimmed" \
    --run_root_dir "$WORKDIR/output/lora" \
    --batch_size 16 \
    --learning_rate 5e-4 \
    --lora_rank 32 \
    --lora_dropout 0.0 \
    --use_lora True \
    --use_quantization False \
    --image_aug True \
    --max_steps 50000 \
    --save_steps 1000 \
    --wandb_project "openvla-piper-lora" \
    --run_id_note "piper_pick"

echo ""
echo "  Training complete!"
echo "  Next: python export_model.py --recipe lora"
