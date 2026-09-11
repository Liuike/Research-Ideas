#!/usr/bin/env bash
#SBATCH --job-name=historical-reproduction
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --array=0-23%2
#SBATCH --output=logs/slurm/%x-%A_%a.out
#SBATCH --error=logs/slurm/%x-%A_%a.err

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}}"
MODE="${1:-scientific}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
cd "$PROJECT_ROOT"
mkdir -p logs/slurm

# One array enforces a shared two-GPU cap across all three scientific groups.
# 0..11 Latch: length, noise, seed; 12..17 Two-Sequence scout: noise, LR;
# 18..23 Parity scout: noise, LR. Scout seed=1000, Latch seeds=0,1,2.
# Smoke overrides the submission array to 0-3%2: Two-Sequence then Parity,
# each noise off/on, seed999. Never alter the actual Slurm task identity.
case "$MODE" in
    scientific)
        if (( TASK_ID >= 0 && TASK_ID < 12 )); then
            CONFIG="configs/historical/latch_initial.yaml"
            PLAN_TASK_ID="$TASK_ID"
        elif (( TASK_ID < 18 && TASK_ID >= 12 )); then
            CONFIG="configs/historical/two_sequence_scout.yaml"
            PLAN_TASK_ID="$((TASK_ID - 12))"
        elif (( TASK_ID < 24 && TASK_ID >= 18 )); then
            CONFIG="configs/historical/parity_scout.yaml"
            PLAN_TASK_ID="$((TASK_ID - 18))"
        else
            echo "Scientific task must be 0..23" >&2; exit 1
        fi
        ;;
    smoke)
        if (( TASK_ID >= 0 && TASK_ID < 2 )); then
            CONFIG="configs/historical/two_sequence_gpu_smoke.yaml"
            PLAN_TASK_ID="$TASK_ID"
        elif (( TASK_ID >= 2 && TASK_ID < 4 )); then
            CONFIG="configs/historical/parity_gpu_smoke.yaml"
            PLAN_TASK_ID="$((TASK_ID - 2))"
        else
            echo "Smoke task must be 0..3" >&2; exit 1
        fi
        ;;
    *) echo "Mode must be scientific or smoke" >&2; exit 1 ;;
esac
export PROJECT_ROOT PLAN_TASK_ID
COMMAND=(bash "${SCRIPT_DIR}/oscar_historical_latch.sbatch" "$CONFIG")
printf -v FINAL_COMMAND '%q ' "${COMMAND[@]}"
echo "Mode: $MODE; Slurm task: $TASK_ID; config task: $PLAN_TASK_ID"
echo "Dispatch command: $FINAL_COMMAND"
"${COMMAND[@]}"
