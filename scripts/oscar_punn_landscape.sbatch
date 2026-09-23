#!/usr/bin/env bash
#SBATCH --job-name=punn-landscape-basic
#SBATCH --partition=batch
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --array=0-179%12
#SBATCH --output=logs/slurm/%x-%A_%a.out
#SBATCH --error=logs/slurm/%x-%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}}"
PLAN_FILE="${1:?Pass a JSONL plan created with experiment plan --plan-file}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK_ID < 0 || TASK_ID >= 180 )); then
  echo "Unknown PUNN task id: ${TASK_ID}" >&2
  exit 1
fi

# Config order is tasks=[f1,f4], methods=[sgd,pso,de], seeds=[0..29].
TASK_INDEX="$((TASK_ID / 90))"
METHOD_INDEX="$(((TASK_ID % 90) / 30))"
SEED="$((TASK_ID % 30))"
TASKS=(f1 f4)
METHODS=(sgd pso de)
TASK="${TASKS[TASK_INDEX]}"
METHOD="${METHODS[METHOD_INDEX]}"
DATA_SEED="$((100000 + SEED))"
LOG_GROUP="punn-landscape-basic-v1"
RUN_NAME="punn-${TASK}-${METHOD}-seed${SEED}"
CONDITION="${TASK}-${METHOD}"

UV_BIN="${UV_BIN:-}"
if [[ -z "${UV_BIN}" && -x "${PROJECT_ROOT}/.uv-bootstrap/bin/uv" ]]; then
  UV_BIN="${PROJECT_ROOT}/.uv-bootstrap/bin/uv"
fi
if [[ -z "${UV_BIN}" ]]; then
  UV_BIN="$(command -v uv || true)"
fi
cd "${PROJECT_ROOT}"
unset LD_LIBRARY_PATH
unset VIRTUAL_ENV
if [[ -z "${UV_BIN}" || ! -x "${UV_BIN}" ]]; then
  echo "uv is unavailable; run scripts/oscar_setup_uv.sh first" >&2
  exit 1
fi
if [[ ! -f "${PLAN_FILE}" ]]; then
  echo "Plan file not found: ${PLAN_FILE}" >&2
  exit 1
fi

mapfile -t COMMAND < <("${UV_BIN}" run --frozen --no-sync python -c \
  'import json,sys; rows=open(sys.argv[1], encoding="utf-8").read().splitlines(); command=json.loads(rows[int(sys.argv[2])]); print("\n".join(command))' \
  "${PLAN_FILE}" "${TASK_ID}" | tr -d '\r')
if (( ${#COMMAND[@]} == 0 )); then
  echo "No plan command for task ${TASK_ID}" >&2
  exit 1
fi
PLAN_TASK="none"
PLAN_METHOD="none"
PLAN_SEED="none"
PLAN_DATA_SEED="none"
PLAN_GROUP="none"
PLAN_NAME="none"
for ((i = 0; i < ${#COMMAND[@]} - 1; i++)); do
  case "${COMMAND[$i]}" in
    --task) PLAN_TASK="${COMMAND[$((i + 1))]}" ;;
    --method) PLAN_METHOD="${COMMAND[$((i + 1))]}" ;;
    --seed) PLAN_SEED="${COMMAND[$((i + 1))]}" ;;
    --data-seed) PLAN_DATA_SEED="${COMMAND[$((i + 1))]}" ;;
    --run-group) PLAN_GROUP="${COMMAND[$((i + 1))]}" ;;
    --run-name) PLAN_NAME="${COMMAND[$((i + 1))]}" ;;
  esac
done
if [[ "${PLAN_TASK}" != "${TASK}" || "${PLAN_METHOD}" != "${METHOD}" ||
      "${PLAN_SEED}" != "${SEED}" || "${PLAN_DATA_SEED}" != "${DATA_SEED}" ||
      "${PLAN_GROUP}" != "${LOG_GROUP}" || "${PLAN_NAME}" != "${RUN_NAME}" ]]; then
  echo "Plan entry ${TASK_ID} disagrees with its registered condition/seed mapping" >&2
  exit 1
fi

RUN_COMMAND=("${UV_BIN}" run --frozen --no-sync "${COMMAND[@]}")
printf -v FINAL_COMMAND "%q " "${RUN_COMMAND[@]}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1"
  echo "Project root: ${PROJECT_ROOT}"
  echo "Plan file: ${PLAN_FILE}"
  echo "Task id: ${TASK_ID}"
  echo "Condition: ${CONDITION}"
  echo "Seed: ${SEED}"
  echo "Data seed: ${DATA_SEED}"
  echo "Log group: ${LOG_GROUP}"
  echo "Run name: ${RUN_NAME}"
  echo "Final command: ${FINAL_COMMAND}"
  exit 0
fi

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
"${RUN_COMMAND[@]}"
