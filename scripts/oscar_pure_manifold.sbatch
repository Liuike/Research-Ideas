#!/usr/bin/env bash
#SBATCH --job-name=pure-manifold-da10
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=36:00:00
#SBATCH --array=0-1%2
#SBATCH --output=logs/slurm/%x-%A_%a.out
#SBATCH --error=logs/slurm/%x-%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PLAN_FILE="${1:?Pass the paired Manifold plan created by experiment plan}"
# Direct mapping: row/task 0 = momentum .95, row/task 1 = momentum 0.
# Shared decoder validates argv and prints semantic metadata in DRY_RUN mode.
# The longer allocation reflects the exact-SVD DA-10 engineering timings.
COMMAND=(bash "${SCRIPT_DIR}/oscar_pure_cifar.sbatch" "${PLAN_FILE}")
"${COMMAND[@]}"
