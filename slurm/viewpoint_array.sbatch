#!/bin/bash
#SBATCH --job-name=viewpoint-ai
#SBATCH --output=slurm/logs/viewpoint_%A_%a.out
#SBATCH --error=slurm/logs/viewpoint_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
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

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "SLURM_ARRAY_TASK_ID is not set. Submit this script with sbatch --array."
    exit 2
fi

if (( SLURM_ARRAY_TASK_ID >= NUM_SHARDS )); then
    echo "Array task ${SLURM_ARRAY_TASK_ID} exceeds NUM_SHARDS=${NUM_SHARDS}."
    exit 2
fi

read -r -a EXTRA_ARGS <<< "${EXPERIMENT_ARGS}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export JAX_PLATFORM_NAME=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "[$(date)] Starting ${SLURM_ARRAY_TASK_ID}/${NUM_SHARDS} -> ${OUTPUT_DIR}"
"${PYTHON_BIN}" viewpoint_inference_experiment.py \
    --output-dir "${OUTPUT_DIR}" \
    --num-shards "${NUM_SHARDS}" \
    --shard-index "${SLURM_ARRAY_TASK_ID}" \
    --no-progress \
    "${EXTRA_ARGS[@]}"
echo "[$(date)] Finished ${SLURM_ARRAY_TASK_ID}/${NUM_SHARDS}"
