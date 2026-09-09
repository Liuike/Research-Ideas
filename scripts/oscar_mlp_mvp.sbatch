#!/usr/bin/env bash
#SBATCH --job-name=resurrect-mlp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --array=0-159
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
UV_BIN="${UV_BIN:-$(command -v uv || true)}"
if [[ -z "${UV_BIN}" && -x "${PROJECT_ROOT}/.uv-bootstrap/bin/uv" ]]; then
  UV_BIN="${PROJECT_ROOT}/.uv-bootstrap/bin/uv"
fi
if [[ -z "${UV_BIN}" || ! -x "${UV_BIN}" ]]; then
  echo "uv is unavailable; provision the locked Oscar environment first" >&2
  exit 1
fi
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
OPTIMIZERS=(adam_o muon_o rsgd mm)
DEPTHS=(4 8 16 32)
ACTIVATIONS=(logistic tanh)
SEEDS_PER_CELL=5
SEED="$((TASK_ID % SEEDS_PER_CELL))"
CELL="$((TASK_ID / SEEDS_PER_CELL))"
OPTIMIZER="${OPTIMIZERS[$((CELL % 4))]}"
CELL="$((CELL / 4))"
DEPTH="${DEPTHS[$((CELL % 4))]}"
ACTIVATION="${ACTIVATIONS[$((CELL / 4))]}"

WIDTH="${WIDTH:-1000}"
STEPS="${STEPS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
case "${OPTIMIZER}" in
  adam_o) LEARNING_RATE="${LR_ADAM_O:-0.001}" ;;
  muon_o) LEARNING_RATE="${LR_MUON_O:-0.001}" ;;
  rsgd) LEARNING_RATE="${LR_RSGD:-0.001}" ;;
  mm) LEARNING_RATE="${LR_MM:-0.001}" ;;
esac
AUX_LEARNING_RATE="${AUX_LEARNING_RATE:-0.001}"
LOG_GROUP="${LOG_GROUP:-mlp-mnist-mvp-pilot}"
RUN_NAME="${ACTIVATION}__depth${DEPTH}__${OPTIMIZER}__seed${SEED}__lr-${LEARNING_RATE}"
COMMAND=(
  "${UV_BIN}" run --frozen --no-sync python -m optimizer_resurrection.train
  --track mlp --dataset mnist --activation "${ACTIVATION}"
  --depth "${DEPTH}" --width "${WIDTH}" --optimizer "${OPTIMIZER}"
  --seed "${SEED}" --data-seed "${SEED}" --steps "${STEPS}"
  --batch-size "${BATCH_SIZE}" --learning-rate "${LEARNING_RATE}"
  --aux-learning-rate "${AUX_LEARNING_RATE}" --weight-decay 0
  --success-loss 0.20 --success-accuracy 0.95
  --device cuda --run-group "${LOG_GROUP}" --run-name "${RUN_NAME}"
  --stage mvp
  --recipe-version "${RECIPE_VERSION:-unfrozen-pilot}"
)
printf -v FINAL_COMMAND "%q " "${COMMAND[@]}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1"
  echo "Project root: ${PROJECT_ROOT}"
  echo "Condition: ${ACTIVATION}/${OPTIMIZER}/depth-${DEPTH}"
  echo "Seed: ${SEED}"
  echo "Task id: ${TASK_ID}"
  echo "Log group: ${LOG_GROUP}"
  echo "Recipe version: ${RECIPE_VERSION:-unfrozen-pilot}"
  echo "Run name: ${RUN_NAME}"
  echo "Final command: ${FINAL_COMMAND}"
  exit 0
fi

cd "${PROJECT_ROOT}"
unset LD_LIBRARY_PATH
unset VIRTUAL_ENV
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
"${COMMAND[@]}"
