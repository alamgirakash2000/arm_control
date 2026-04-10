#!/bin/bash
# Local setup for RTX 4090 — creates conda env + converts data
set -e

WORKDIR=$(cd "$(dirname "$0")" && pwd)
cd "$WORKDIR"

echo "============================================"
echo "  OpenVLA Local Setup (RTX 4090)"
echo "============================================"
echo ""

# ── 1. Create conda environment ────────────────────────────────
echo "[1/6] Creating conda environment: openvla ..."
conda create -n openvla python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate openvla

# ── 2. Install PyTorch with CUDA ───────────────────────────────
echo "[2/6] Installing PyTorch 2.2.0 + CUDA 12.4 (pinned for OpenVLA) ..."
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu124

# ── 3. Clone and install OpenVLA ───────────────────────────────
echo "[3/6] Cloning OpenVLA ..."
if [ ! -d "openvla" ]; then
    git clone https://github.com/openvla/openvla.git
fi
cd openvla
pip install -e .
cd "$WORKDIR"

# ── 4. Install flash-attn + extras ────────────────────────────
echo "[4/6] Installing flash-attention + dependencies ..."
pip install packaging ninja
pip install "flash-attn==2.5.5" --no-build-isolation 2>/dev/null || \
    echo "  WARNING: flash-attn failed. Training works without it but slower."
pip install bitsandbytes peft accelerate wandb
pip install zarr numcodecs tensorflow tensorflow-datasets==4.9.3 opencv-python-headless

# ── 5. Convert Zarr to RLDS ───────────────────────────────────
echo "[5/6] Converting Zarr data to RLDS ..."
FOUND_DATA=0
for dataset_dir in data/*/; do
    if [ -d "${dataset_dir}dataset.zarr" ]; then
        dataset_name=$(basename "$dataset_dir")
        echo "  Converting: $dataset_name"
        python convert_zarr_to_rlds.py \
            --zarr_dir "${dataset_dir}" \
            --output_dir "rlds_data/${dataset_name}" \
            --task_instruction "pick the object from the table"
        FOUND_DATA=1
    fi
done

if [ $FOUND_DATA -eq 0 ]; then
    echo "  ERROR: No Zarr datasets found in data/"
    echo "  Run: cp -r ../data/split/pick_trimmed data/pick_trimmed"
    exit 1
fi

# ── 6. Register dataset ──────────────────────────────────────
echo "[6/6] Registering dataset with OpenVLA ..."
python register_dataset.py

echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "  To train:"
echo "    conda activate openvla"
echo "    bash run_local.sh"
echo ""
