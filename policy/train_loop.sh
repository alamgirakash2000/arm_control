#!/bin/bash
# Watchdog-based training supervisor.
# Restarts on clean auto-restart, crashes, or prolonged stalls with no progress updates.
#
# Usage:
#   bash policy/train_loop.sh ./data/pick ./checkpoints/pick
#   bash policy/train_loop.sh ./data/pick ./checkpoints/pick 500
#   bash policy/train_loop.sh ./data/pick ./checkpoints/pick 500 20 --batch_size 16
#
# Optional env vars:
#   CONDA_ENV=armcontrol-train-gpu-wheel-py310-cu124  # conda env to activate
#   VERIFY_CUDA=1            # fail fast if torch.cuda.is_available() is false
#   CUDA_LAUNCH_BLOCKING=1   # optional debugging override
#   STALL_TIMEOUT_SEC=1200   # restart if no status update for 20 minutes
#   POLL_INTERVAL_SEC=30     # watchdog polling interval
#   RESTART_DELAY_SEC=5      # delay before relaunch
#   HARD_KILL_AFTER_SEC=30   # wait after SIGTERM before SIGKILL

set -u

DATASET_DIR="${1:?Usage: bash train_loop.sh <dataset_dir> <output_dir> [epochs] [epochs_per_run] [extra train args...]}"
OUTPUT_DIR="${2:?Usage: bash train_loop.sh <dataset_dir> <output_dir> [epochs] [epochs_per_run] [extra train args...]}"
shift 2

TOTAL_EPOCHS="500"
EPOCHS_PER_RUN="0"
if [ "$#" -gt 0 ] && [[ "$1" != -* ]]; then
    TOTAL_EPOCHS="$1"
    shift
fi
if [ "$#" -gt 0 ] && [[ "$1" != -* ]]; then
    EPOCHS_PER_RUN="$1"
    shift
fi
EXTRA_ARGS=("$@")

STALL_TIMEOUT_SEC="${STALL_TIMEOUT_SEC:-1200}"
POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC:-30}"
RESTART_DELAY_SEC="${RESTART_DELAY_SEC:-5}"
HARD_KILL_AFTER_SEC="${HARD_KILL_AFTER_SEC:-30}"
STARTUP_GRACE_SEC="${STARTUP_GRACE_SEC:-180}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Activate conda env if not already active
CONDA_ENV="${CONDA_ENV:-armcontrol-train-gpu-wheel-py310-cu124}"
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV" 2>/dev/null || {
        echo "Failed to activate conda env: $CONDA_ENV" >&2
        exit 1
    }
fi
PYTHON_BIN="$(command -v python3 || command -v python)"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export DNNL_MAX_CPU_ISA="${DNNL_MAX_CPU_ISA:-AVX2}"
export ONEDNN_MAX_CPU_ISA="${ONEDNN_MAX_CPU_ISA:-AVX2}"
export MKL_ENABLE_INSTRUCTIONS="${MKL_ENABLE_INSTRUCTIONS:-AVX2}"

if [ "${VERIFY_CUDA:-1}" = "1" ]; then
    "$PYTHON_BIN" - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    sys.stderr.write("CUDA is not available in the active training environment.\n")
    sys.stderr.write(
        f"torch={torch.__version__}, torch.version.cuda={torch.version.cuda}, "
        f"cuda_built={torch.backends.cuda.is_built()}\n"
    )
    raise SystemExit(1)
PY
fi

if [ "${CUDA_LAUNCH_BLOCKING:-0}" = "1" ]; then
    export CUDA_LAUNCH_BLOCKING=1
else
    unset CUDA_LAUNCH_BLOCKING 2>/dev/null || true
fi

STATUS_FILE="$OUTPUT_DIR/training_status.json"
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"

RUN=1
TRAIN_PID=""
BASE_SEED=42
RUN_START_TS=0

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

log() {
    printf '[%s] %s\n' "$(timestamp)" "$1" | tee -a "$LOG_FILE"
}

cleanup_child() {
    if [ -n "${TRAIN_PID:-}" ] && kill -0 "$TRAIN_PID" 2>/dev/null; then
        kill -TERM "$TRAIN_PID" 2>/dev/null || true
        wait "$TRAIN_PID" 2>/dev/null || true
    fi
}

trap 'cleanup_child; exit 130' INT TERM

launch_training() {
    local run_seed=$((BASE_SEED + RUN - 1))
    RUN_START_TS=$(date +%s)
    rm -f "$STATUS_FILE"
    local cmd=(
        "$PYTHON_BIN" -u "$SCRIPT_DIR/train.py"
        --dataset_dir "$DATASET_DIR"
        --output_dir "$OUTPUT_DIR"
        --epochs "$TOTAL_EPOCHS"
        --epochs_per_run "$EPOCHS_PER_RUN"
        --resume latest
        --seed "$run_seed"
    )

    if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
        cmd+=("${EXTRA_ARGS[@]}")
    fi

    if command -v ionice >/dev/null 2>&1; then
        cmd=(ionice -c2 -n7 "${cmd[@]}")
    fi
    if command -v nice >/dev/null 2>&1; then
        cmd=(nice -n 10 "${cmd[@]}")
    fi

    "${cmd[@]}" \
        > >(tee -a "$LOG_FILE") \
        2> >(tee -a "$LOG_FILE" >&2) &
    TRAIN_PID=$!
}

wait_for_exit_or_stall() {
    while kill -0 "$TRAIN_PID" 2>/dev/null; do
        local heartbeat=0
        local now
        now=$(date +%s)

        if [ -f "$STATUS_FILE" ]; then
            heartbeat=$(stat -c %Y "$STATUS_FILE")
        elif [ -f "$LOG_FILE" ]; then
            heartbeat=$(stat -c %Y "$LOG_FILE")
        fi

        if [ "$heartbeat" -lt "$RUN_START_TS" ]; then
            heartbeat="$RUN_START_TS"
        fi

        if [ "$heartbeat" -gt 0 ]; then
            local idle
            idle=$((now - heartbeat))
            if [ "$idle" -ge "$STALL_TIMEOUT_SEC" ] && [ $((now - RUN_START_TS)) -ge "$STARTUP_GRACE_SEC" ]; then
                log "Watchdog: no progress update for ${idle}s. Restarting training process."
                kill -TERM "$TRAIN_PID" 2>/dev/null || true
                local waited=0
                while kill -0 "$TRAIN_PID" 2>/dev/null && [ "$waited" -lt "$HARD_KILL_AFTER_SEC" ]; do
                    sleep 1
                    waited=$((waited + 1))
                done
                if kill -0 "$TRAIN_PID" 2>/dev/null; then
                    log "Watchdog: process ignored SIGTERM. Sending SIGKILL."
                    kill -KILL "$TRAIN_PID" 2>/dev/null || true
                fi
                wait "$TRAIN_PID" 2>/dev/null || true
                return 124
            fi
        fi

        sleep "$POLL_INTERVAL_SEC"
    done

    wait "$TRAIN_PID"
    return $?
}

log "========================================="
log "Training Supervisor"
log "Dataset:        $DATASET_DIR"
log "Output:         $OUTPUT_DIR"
log "Total epochs:   $TOTAL_EPOCHS"
log "Epochs per run: $EPOCHS_PER_RUN"
log "Stall timeout:  ${STALL_TIMEOUT_SEC}s"
log "Startup grace:  ${STARTUP_GRACE_SEC}s"
log "Log file:       $LOG_FILE"
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    log "Extra args:     ${EXTRA_ARGS[*]}"
fi
log "========================================="

while true; do
    log "--- Run #$RUN (seed $((BASE_SEED + RUN - 1))) ---"
    launch_training
    wait_for_exit_or_stall
    EXIT_CODE=$?
    TRAIN_PID=""

    if [ "$EXIT_CODE" -eq 42 ]; then
        log "Trainer requested clean restart. Relaunching in ${RESTART_DELAY_SEC}s."
        sleep "$RESTART_DELAY_SEC"
        RUN=$((RUN + 1))
        continue
    fi

    if [ "$EXIT_CODE" -eq 0 ]; then
        log "Training complete."
        break
    fi

    if [ "$EXIT_CODE" -eq 124 ]; then
        log "Watchdog restart completed. Relaunching in ${RESTART_DELAY_SEC}s."
    else
        log "Trainer exited with code $EXIT_CODE. Relaunching in ${RESTART_DELAY_SEC}s."
    fi
    sleep "$RESTART_DELAY_SEC"
    RUN=$((RUN + 1))
done
