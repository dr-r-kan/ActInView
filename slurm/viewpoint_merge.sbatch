#!/bin/bash
#SBATCH --job-name=viewpoint-merge
#SBATCH --output=slurm/logs/viewpoint_merge_%j.out
#SBATCH --error=slurm/logs/viewpoint_merge_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=04:00:00

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
mkdir -p slurm/logs

NUM_SHARDS="${NUM_SHARDS:-160}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs_viewpoint_pymdp_study1_array}"
PYTHON_BIN="${PYTHON_BIN:-python}"
EXPERIMENT_ARGS="${EXPERIMENT_ARGS:-}"

read -r -a EXTRA_ARGS <<< "${EXPERIMENT_ARGS}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export JAX_PLATFORM_NAME=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "[$(date)] Merging ${NUM_SHARDS} shard(s) from ${OUTPUT_DIR}"
"${PYTHON_BIN}" viewpoint_inference_experiment.py \
    --output-dir "${OUTPUT_DIR}" \
    --num-shards "${NUM_SHARDS}" \
    --merge-shards \
    --no-progress \
    "${EXTRA_ARGS[@]}"
echo "[$(date)] Merge finished"
