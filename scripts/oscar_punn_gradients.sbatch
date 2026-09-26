#!/usr/bin/env bash
#SBATCH --job-name=punn-gradients
#SBATCH --partition=batch
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --array=0-79%12
#SBATCH --output=logs/slurm/%x-%A_%a.out
#SBATCH --error=logs/slurm/%x-%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}}"
PLAN_FILE="${1:?Pass a JSONL plan created with experiment plan --plan-file}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

cd "${PROJECT_ROOT}"
unset LD_LIBRARY_PATH
unset VIRTUAL_ENV
export UV_CACHE_DIR="${UV_CACHE_DIR:-${PROJECT_ROOT}/.cache/uv}"
export WANDB_DIR="${WANDB_DIR:-${PROJECT_ROOT}/wandb}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-${PROJECT_ROOT}/wandb}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${PROJECT_ROOT}/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${PROJECT_ROOT}/wandb}"
if [[ "${PLAN_FILE}" != /* ]]; then
  PLAN_FILE="${PROJECT_ROOT}/${PLAN_FILE}"
fi
if [[ ! -f "${PLAN_FILE}" ]]; then
  echo "Plan file not found: ${PLAN_FILE}" >&2
  exit 1
fi

UV_BIN="${UV_BIN:-}"
if [[ -z "${UV_BIN}" && -x "${PROJECT_ROOT}/.uv-bootstrap/bin/uv" ]]; then
  UV_BIN="${PROJECT_ROOT}/.uv-bootstrap/bin/uv"
fi
if [[ -z "${UV_BIN}" ]]; then
  UV_BIN="$(command -v uv || true)"
fi
if [[ -z "${UV_BIN}" || ! -x "${UV_BIN}" ]]; then
  echo "uv is unavailable; provision the locked Oscar environment first" >&2
  exit 1
fi

# Plan line N maps directly to array task N-1. Decode the JSONL with the locked
# interpreter, then run the planned workload with the same frozen environment.
COMMAND_FILE="$(mktemp)"
trap 'rm -f -- "${COMMAND_FILE}"' EXIT
"${UV_BIN}" run --frozen --no-sync python - "${PLAN_FILE}" "${TASK_ID}" "${COMMAND_FILE}" <<'PY'
import json
import sys

plan_path, task_id_text, output_path = sys.argv[1:]
try:
    task_id = int(task_id_text)
except ValueError as exc:
    raise SystemExit(f"invalid array task id: {task_id_text!r}") from exc
if task_id < 0:
    raise SystemExit(f"array task id must be nonnegative: {task_id}")

with open(plan_path, encoding="utf-8") as plan_file:
    rows = [line for line in plan_file if line.strip()]
if task_id >= len(rows):
    raise SystemExit(
        f"array task id {task_id} is outside the {len(rows)}-command plan"
    )
try:
    command = json.loads(rows[task_id])
except json.JSONDecodeError as exc:
    raise SystemExit(f"invalid JSON on plan line {task_id + 1}: {exc}") from exc
if not isinstance(command, list) or not command or not all(
    isinstance(item, str) and "\0" not in item for item in command
):
    raise SystemExit(f"plan line {task_id + 1} must be a nonempty string argv list")
with open(output_path, "wb") as output_file:
    output_file.write(b"\0".join(item.encode() for item in command) + b"\0")
PY

mapfile -d '' -t PLAN_COMMAND < "${COMMAND_FILE}"
if (( ${#PLAN_COMMAND[@]} == 0 )); then
  echo "No command for task ${TASK_ID}" >&2
  exit 1
fi

TASK="none"
METHOD="none"
SEED="none"
DATA_SEED="none"
STAGE="none"
LOG_GROUP="none"
RUN_NAME="none"
for ((index = 0; index < ${#PLAN_COMMAND[@]} - 1; index++)); do
  case "${PLAN_COMMAND[$index]}" in
    --task) TASK="${PLAN_COMMAND[$((index + 1))]}" ;;
    --method) METHOD="${PLAN_COMMAND[$((index + 1))]}" ;;
    --seed) SEED="${PLAN_COMMAND[$((index + 1))]}" ;;
    --data-seed) DATA_SEED="${PLAN_COMMAND[$((index + 1))]}" ;;
    --stage) STAGE="${PLAN_COMMAND[$((index + 1))]}" ;;
    --run-group) LOG_GROUP="${PLAN_COMMAND[$((index + 1))]}" ;;
    --run-name) RUN_NAME="${PLAN_COMMAND[$((index + 1))]}" ;;
  esac
done
if [[ "${TASK}" == "none" || "${METHOD}" == "none" || "${SEED}" == "none" || \
      "${DATA_SEED}" == "none" || "${STAGE}" == "none" || \
      "${LOG_GROUP}" == "none" || "${RUN_NAME}" == "none" ]]; then
  echo "Plan command lacks task/method/seed/data-seed/stage/run-group/run-name metadata" >&2
  exit 1
fi

COMMAND=("${UV_BIN}" run --frozen --no-sync "${PLAN_COMMAND[@]}")
printf -v FINAL_COMMAND "%q " "${COMMAND[@]}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1"
  echo "Project root: ${PROJECT_ROOT}"
  echo "Plan file: ${PLAN_FILE}"
  echo "Task id: ${TASK_ID}"
  echo "Plan line: $((TASK_ID + 1))"
  echo "Task: ${TASK}"
  echo "Method: ${METHOD}"
  echo "Seed: ${SEED}"
  echo "Data seed: ${DATA_SEED}"
  echo "Stage: ${STAGE}"
  echo "Log group: ${LOG_GROUP}"
  echo "Run name: ${RUN_NAME}"
  echo "Final command: ${FINAL_COMMAND}"
  exit 0
fi

mkdir -p "${PROJECT_ROOT}/logs/slurm"
mkdir -p "${PROJECT_ROOT}/wandb"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
"${COMMAND[@]}"
