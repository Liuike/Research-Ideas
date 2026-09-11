#!/usr/bin/env bash
#SBATCH --job-name=historical-latch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --array=0-11%2
#SBATCH --output=logs/slurm/%x-%A_%a.out
#SBATCH --error=logs/slurm/%x-%A_%a.err

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}}"
CONFIG="${1:?Pass committed historical Latch YAML}"
TASK_ID="${PLAN_TASK_ID:-${SLURM_ARRAY_TASK_ID:-0}}"
UV_BIN="${UV_BIN:-${PROJECT_ROOT}/.uv-bootstrap/bin/uv}"
cd "$PROJECT_ROOT"
if [[ -n "${EXPECTED_REVISION:-}" && "$(git rev-parse HEAD)" != "$EXPECTED_REVISION" ]]; then
    echo "Checkout no longer matches the submitted revision" >&2
    exit 1
fi
unset LD_LIBRARY_PATH
unset VIRTUAL_ENV
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=2

# The frozen planner orders length, noise, seed, then learning rate. Evaluation
# tasks 0..2: T10/noise0, 3..5: T10/noise.2, 6..8: T100/noise0,
# 9..11: T100/noise.2; each triplet uses direct seeds 0,1,2.
# Smoke and scout use their own committed axes and explicit submission arrays.
mapfile -t PLANNED < <("$UV_BIN" run --frozen --no-sync python -m \
    optimizer_resurrection.experiment plan "$CONFIG" | tr -d '\r')
if (( TASK_ID < 0 || TASK_ID >= ${#PLANNED[@]} - 1 )); then
    echo "Task id outside registered plan: $TASK_ID" >&2
    exit 1
fi
mapfile -t RUN_ARGS < <("$UV_BIN" run --frozen --no-sync python -c \
    'import json,sys; print("\n".join(json.loads(sys.argv[1])))' "${PLANNED[$TASK_ID]}")
if (( ${#RUN_ARGS[@]} < 3 )); then
    echo "Planner did not return a training command" >&2
    exit 1
fi
SEED="unknown"
LOG_GROUP="unknown"
RUN_NAME="unknown"
for ((i=0; i<${#RUN_ARGS[@]}-1; i++)); do
    case "${RUN_ARGS[$i]}" in
        --seed) SEED="${RUN_ARGS[$((i+1))]}" ;;
        --run-group) LOG_GROUP="${RUN_ARGS[$((i+1))]}" ;;
        --run-name) RUN_NAME="${RUN_ARGS[$((i+1))]}" ;;
    esac
done
COMMAND=("$UV_BIN" run --frozen --no-sync "${RUN_ARGS[@]}")
printf -v FINAL_COMMAND '%q ' "${COMMAND[@]}"
echo "Project root: $PROJECT_ROOT"
echo "Config: $CONFIG"
echo "Expected revision: ${EXPECTED_REVISION:-not-set}"
echo "Task id: $TASK_ID"
echo "Seed: $SEED"
echo "Log group: $LOG_GROUP"
echo "Run name: $RUN_NAME"
echo "Final command: $FINAL_COMMAND"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY_RUN=1"
    exit 0
fi
"${COMMAND[@]}"
