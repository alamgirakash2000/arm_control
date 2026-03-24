#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_NAME="${CONDA_ENV:-armcontrol-train-gpu-wheel-py310-cu124}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
    echo "conda.sh not found under \$HOME/miniconda3" >&2
    exit 1
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export DNNL_MAX_CPU_ISA="${DNNL_MAX_CPU_ISA:-AVX2}"
export ONEDNN_MAX_CPU_ISA="${ONEDNN_MAX_CPU_ISA:-AVX2}"
export MKL_ENABLE_INSTRUCTIONS="${MKL_ENABLE_INSTRUCTIONS:-AVX2}"

if [ "$REQUIRE_CUDA" = "1" ]; then
    conda run --no-capture-output -n "$ENV_NAME" python - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    sys.stderr.write("CUDA is not available in the active environment.\n")
    sys.stderr.write(
        f"torch={torch.__version__}, torch.version.cuda={torch.version.cuda}, "
        f"cuda_built={torch.backends.cuda.is_built()}\n"
    )
    raise SystemExit(1)
PY
fi

cd "$SCRIPT_DIR/.."
exec conda run --no-capture-output -n "$ENV_NAME" python policy/train.py "$@"
