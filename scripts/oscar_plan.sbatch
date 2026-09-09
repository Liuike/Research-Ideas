#!/usr/bin/env bash
#SBATCH --job-name=resurrection-plan
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --array=0-0
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}}"
PLAN_FILE="${1:?Pass a JSONL plan created with experiment plan --plan-file}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
PLAN_LINE="$((TASK_ID + 1))"
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
  echo "uv is unavailable; provision the locked Oscar environment first" >&2
  exit 1
fi

if [[ ! -f "${PLAN_FILE}" ]]; then
  echo "Plan file not found: ${PLAN_FILE}" >&2
  exit 1
fi

mapfile -t COMMAND < <("${UV_BIN}" run --frozen --no-sync python -c 'import json,sys; rows=open(sys.argv[1], encoding="utf-8").read().splitlines(); command=json.loads(rows[int(sys.argv[2])]); print("\n".join(command))' "${PLAN_FILE}" "${TASK_ID}" | tr -d '\r')
if (( ${#COMMAND[@]} == 0 )); then
  echo "No command for task ${TASK_ID} (line ${PLAN_LINE})" >&2
  exit 1
fi
LOG_GROUP="none"
RUN_NAME="none"
STAGE="none"
RECIPE_VERSION="none"
for ((i = 0; i < ${#COMMAND[@]} - 1; i++)); do
  case "${COMMAND[$i]}" in
    --run-group) LOG_GROUP="${COMMAND[$((i + 1))]}" ;;
    --run-name) RUN_NAME="${COMMAND[$((i + 1))]}" ;;
    --stage) STAGE="${COMMAND[$((i + 1))]}" ;;
    --recipe-version) RECIPE_VERSION="${COMMAND[$((i + 1))]}" ;;
  esac
done
RUN_COMMAND=("${UV_BIN}" run --frozen --no-sync "${COMMAND[@]}")
printf -v FINAL_COMMAND "%q " "${RUN_COMMAND[@]}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1"
  echo "Project root: ${PROJECT_ROOT}"
  echo "Plan file: ${PLAN_FILE}"
  echo "Task id: ${TASK_ID}"
  echo "Plan line: ${PLAN_LINE}"
  echo "Stage: ${STAGE}"
  echo "Log group: ${LOG_GROUP}"
  echo "Run name: ${RUN_NAME}"
  echo "Recipe version: ${RECIPE_VERSION}"
  echo "Final command: ${FINAL_COMMAND}"
  exit 0
fi

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
"${RUN_COMMAND[@]}"
