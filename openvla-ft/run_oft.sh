#!/bin/bash
# Cloud OFT fine-tuning: 4-8x A100 80GB, ~1-2 days, ~$150-500
# Produces 25-50x faster inference than LoRA
set -e

WORKDIR=$(cd "$(dirname "$0")" && pwd)

# Clone OFT repo if not present
if [ ! -d "$WORKDIR/openvla-oft" ]; then
    echo "Cloning OpenVLA-OFT repository..."
    cd "$WORKDIR"
    git clone https://github.com/moojink/openvla-oft.git
    cd openvla-oft
    pip install -e .
fi

cd "$WORKDIR/openvla-oft"

NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "============================================"
echo "  OpenVLA-OFT Fine-Tuning (CLOUD)"
echo "============================================"
echo "  GPUs: ${NUM_GPUS}x A100 80GB"
echo "  Estimated time: 1-2 days"
echo "============================================"
echo ""

torchrun --standalone --nnodes 1 --nproc-per-node "$NUM_GPUS" \
    vla-scripts/finetune.py \
    --vla_path "openvla/openvla-7b" \
    --data_root_dir "$WORKDIR/data/pick_trimmed" \
    --dataset_name "piper_pick_trimmed" \
    --run_root_dir "$WORKDIR/output/oft" \
    --batch_size 8 \
    --learning_rate 5e-4 \
    --lora_rank 32 \
    --use_lora True \
    --use_l1_regression True \
    --use_film True \
    --use_proprio True \
    --center_crop True \
    --image_aug True \
    --max_steps 150000 \
    --save_steps 1000 \
    --wandb_project "openvla-piper-oft" \
    --run_id_note "piper_pick_oft"

echo ""
echo "  Training complete!"
echo "  Next: python export_model.py --recipe oft"
