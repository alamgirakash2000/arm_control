#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="$SCRIPT_DIR/environment.train-stable.yml"
ENV_NAME="${1:-armcontrol-train-gpu-wheel-py310-cu124}"

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
    echo "conda.sh not found under \$HOME/miniconda3" >&2
    exit 1
fi

if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    conda env update --file "$ENV_FILE" --name "$ENV_NAME" --prune
else
    conda env create --file "$ENV_FILE" --name "$ENV_NAME"
fi

echo "Environment ready: $ENV_NAME"
echo "Activate it with: conda activate $ENV_NAME"
