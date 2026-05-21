#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NUM_SHARDS="${NUM_SHARDS:-160}"
MAX_CONCURRENT="${MAX_CONCURRENT:-80}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs_viewpoint_pymdp_study1_array}"
PYTHON_BIN="${PYTHON_BIN:-python}"
EXPERIMENT_ARGS="${EXPERIMENT_ARGS:-}"

if (( NUM_SHARDS < 1 )); then
    echo "NUM_SHARDS must be at least 1."
    exit 2
fi

if (( MAX_CONCURRENT < 1 )); then
    echo "MAX_CONCURRENT must be at least 1."
    exit 2
fi

array_job_id="$(
    sbatch --parsable \
        --array="0-$((NUM_SHARDS - 1))%${MAX_CONCURRENT}" \
        --export=ALL,NUM_SHARDS="${NUM_SHARDS}",OUTPUT_DIR="${OUTPUT_DIR}",PYTHON_BIN="${PYTHON_BIN}",EXPERIMENT_ARGS="${EXPERIMENT_ARGS}" \
        "${SCRIPT_DIR}/viewpoint_array.sbatch"
)"

merge_job_id="$(
    sbatch --parsable \
        --dependency="afterok:${array_job_id}" \
        --export=ALL,NUM_SHARDS="${NUM_SHARDS}",OUTPUT_DIR="${OUTPUT_DIR}",PYTHON_BIN="${PYTHON_BIN}",EXPERIMENT_ARGS="${EXPERIMENT_ARGS}" \
        "${SCRIPT_DIR}/viewpoint_merge.sbatch"
)"

echo "Submitted array job: ${array_job_id}"
echo "Submitted merge job: ${merge_job_id} (afterok:${array_job_id})"
echo "Output directory: ${OUTPUT_DIR}"
if [[ -n "${EXPERIMENT_ARGS}" ]]; then
    echo "Extra experiment args: ${EXPERIMENT_ARGS}"
fi
