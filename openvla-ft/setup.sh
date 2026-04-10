#!/bin/bash
# Cloud setup (RunPod/Lambda) — installs deps, clones OpenVLA, prepares data
set -e

WORKDIR=$(cd "$(dirname "$0")" && pwd)
cd "$WORKDIR"

echo "============================================"
echo "  OpenVLA Cloud Setup"
echo "============================================"
echo ""

# ── 1. Install deps ────────────────────────────────────────────
echo "[1/4] Installing dependencies..."
pip install --upgrade pip
pip install packaging ninja
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121

# ── 2. Clone and install OpenVLA ───────────────────────────────
echo "[2/4] Cloning OpenVLA ..."
if [ ! -d "openvla" ]; then
    git clone https://github.com/openvla/openvla.git
fi
cd openvla
pip install -e .
cd "$WORKDIR"

# ── 3. Install extras ─────────────────────────────────────────
echo "[3/4] Installing flash-attention + extras ..."
pip install "flash-attn==2.5.5" --no-build-isolation 2>/dev/null || \
    echo "  WARNING: flash-attn failed. Training works without it but slower."
pip install peft bitsandbytes "accelerate==0.30.0" wandb
pip install zarr numcodecs "numpy<2.0" opencv-python-headless

# ── 4. Verify ─────────────────────────────────────────────────
echo "[4/4] Verifying setup ..."
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}')"
python -c "from prismatic.vla.action_tokenizer import ActionTokenizer; print('OpenVLA OK')"

echo ""
echo "============================================"
echo "  Cloud setup complete!"
echo "============================================"
echo "  Next: bash run_lora.sh  OR  bash run_oft.sh"
echo ""
