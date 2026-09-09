#!/usr/bin/env bash
#SBATCH --job-name=resurrect-select
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SCOUT_CONFIG="${SCOUT_CONFIG:-configs/exploratory/lr_scout_2xl40s.yaml}"
SCOUT_GROUP="${SCOUT_GROUP:-exploratory-2xl40s-lr-scout-v1}"
RECIPE_FILE="${RECIPE_FILE:-plans/exploratory-recipes.json}"
BOUNDARY_PLAN="${BOUNDARY_PLAN:-plans/exploratory-wave2.jsonl}"

SELECT_COMMAND=(
  python -m optimizer_resurrection.experiment select-recipes
  --config "${SCOUT_CONFIG}"
  --run-group "${SCOUT_GROUP}"
  --output "${RECIPE_FILE}"
)
PLAN_COMMAND=(
  python -m optimizer_resurrection.experiment plan
  configs/exploratory/boundary_mlp_2xl40s.yaml
  configs/exploratory/boundary_rnn_2xl40s.yaml
  --recipes "${RECIPE_FILE}"
  --plan-file "${BOUNDARY_PLAN}"
)
printf -v FINAL_SELECT_COMMAND "%q " "${SELECT_COMMAND[@]}"
printf -v FINAL_PLAN_COMMAND "%q " "${PLAN_COMMAND[@]}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1"
  echo "Project root: ${PROJECT_ROOT}"
  echo "Scout group: ${SCOUT_GROUP}"
  echo "Recipe file: ${RECIPE_FILE}"
  echo "Boundary plan: ${BOUNDARY_PLAN}"
  echo "Select command: ${FINAL_SELECT_COMMAND}"
  echo "Plan command: ${FINAL_PLAN_COMMAND}"
  exit 0
fi

cd "${PROJECT_ROOT}"
unset LD_LIBRARY_PATH
unset VIRTUAL_ENV
if [[ -f ".secrets/env" ]]; then
  while IFS='=' read -r key value; do
    [[ -z "${key}" || "${key}" == \#* || -z "${value}" ]] && continue
    if [[ -z "${!key:-}" ]]; then printf -v "${key}" '%s' "${value}"; export "${key}"; fi
  done < ".secrets/env"
fi
source "${VENV_ACTIVATE:-.venv/bin/activate}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "$(dirname -- "${RECIPE_FILE}")"
"${SELECT_COMMAND[@]}"
"${PLAN_COMMAND[@]}"
