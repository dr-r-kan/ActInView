#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

NUM_SHARDS="${NUM_SHARDS:-160}"
MAX_CONCURRENT="${MAX_CONCURRENT:-80}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs_viewpoint_pymdp_study1_array}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-${SCRIPT_DIR}/slurm/logs}"
LEGACY_EXPERIMENT_ARGS="${EXPERIMENT_ARGS:-}"
RUN_ID="${RUN_ID:-viewpoint_$(date -u +%Y%m%dT%H%M%SZ)_$$}"

if (( NUM_SHARDS < 1 )); then
    echo "NUM_SHARDS must be at least 1."
    exit 2
fi

if (( MAX_CONCURRENT < 1 )); then
    echo "MAX_CONCURRENT must be at least 1."
    exit 2
fi

if (( MAX_CONCURRENT > NUM_SHARDS )); then
    MAX_CONCURRENT="${NUM_SHARDS}"
fi

mkdir -p "${OUTPUT_DIR}" "${SLURM_LOG_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd)"
RUN_DIR="${OUTPUT_DIR}/run_logs/${RUN_ID}"
TASK_STATUS_DIR="${RUN_DIR}/tasks"
TASK_LOG_DIR="${RUN_DIR}/task_logs"
MERGE_STATUS_FILE="${RUN_DIR}/merge.status"
MERGE_LOG="${RUN_DIR}/merge.log"
FAILURE_SUMMARY_FILE="${RUN_DIR}/failure_summary.txt"
ARGS_FILE="${RUN_DIR}/experiment_args.txt"
SUBMISSION_FILE="${RUN_DIR}/submission.env"

mkdir -p "${RUN_DIR}" "${TASK_STATUS_DIR}" "${TASK_LOG_DIR}"

EXTRA_ARGS=()
if [[ -n "${LEGACY_EXPERIMENT_ARGS}" ]]; then
    read -r -a LEGACY_ARGS <<< "${LEGACY_EXPERIMENT_ARGS}"
    EXTRA_ARGS+=("${LEGACY_ARGS[@]}")
fi
if (( $# > 0 )); then
    EXTRA_ARGS+=("$@")
fi

: > "${ARGS_FILE}"
if (( ${#EXTRA_ARGS[@]} > 0 )); then
    printf '%s\n' "${EXTRA_ARGS[@]}" > "${ARGS_FILE}"
fi

submitted_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
{
    printf 'run_id=%q\n' "${RUN_ID}"
    printf 'submitted_at=%q\n' "${submitted_at}"
    printf 'project_root=%q\n' "${SCRIPT_DIR}"
    printf 'output_dir=%q\n' "${OUTPUT_DIR}"
    printf 'num_shards=%q\n' "${NUM_SHARDS}"
    printf 'max_concurrent=%q\n' "${MAX_CONCURRENT}"
    printf 'python_bin=%q\n' "${PYTHON_BIN}"
    printf 'slurm_log_dir=%q\n' "${SLURM_LOG_DIR}"
    printf 'task_status_dir=%q\n' "${TASK_STATUS_DIR}"
    printf 'task_log_dir=%q\n' "${TASK_LOG_DIR}"
    printf 'merge_log=%q\n' "${MERGE_LOG}"
    printf 'merge_status_file=%q\n' "${MERGE_STATUS_FILE}"
    printf 'failure_summary_file=%q\n' "${FAILURE_SUMMARY_FILE}"
    printf 'experiment_args_file=%q\n' "${ARGS_FILE}"
    printf 'experiment_arg_count=%q\n' "${#EXTRA_ARGS[@]}"
    printf 'submission_host=%q\n' "$(hostname)"
    printf 'submission_cwd=%q\n' "${PWD}"
} > "${SUBMISSION_FILE}"

array_job_id_raw="$(
    sbatch --parsable \
        --chdir="${SCRIPT_DIR}" \
        --array="0-$((NUM_SHARDS - 1))%${MAX_CONCURRENT}" \
        --output="${SLURM_LOG_DIR}/viewpoint_%A_%a.out" \
        --error="${SLURM_LOG_DIR}/viewpoint_%A_%a.err" \
        --export=ALL,PROJECT_ROOT="${SCRIPT_DIR}",NUM_SHARDS="${NUM_SHARDS}",OUTPUT_DIR="${OUTPUT_DIR}",PYTHON_BIN="${PYTHON_BIN}",RUN_ID="${RUN_ID}",RUN_DIR="${RUN_DIR}",TASK_STATUS_DIR="${TASK_STATUS_DIR}",TASK_LOG_DIR="${TASK_LOG_DIR}",EXPERIMENT_ARGS_FILE="${ARGS_FILE}" \
        "${SCRIPT_DIR}/viewpoint_array.sbatch"
)"
array_job_id="${array_job_id_raw%%;*}"

merge_job_id_raw="$(
    sbatch --parsable \
        --chdir="${SCRIPT_DIR}" \
        --dependency="afterany:${array_job_id}" \
        --output="${SLURM_LOG_DIR}/viewpoint_merge_%j.out" \
        --error="${SLURM_LOG_DIR}/viewpoint_merge_%j.err" \
        --export=ALL,PROJECT_ROOT="${SCRIPT_DIR}",NUM_SHARDS="${NUM_SHARDS}",OUTPUT_DIR="${OUTPUT_DIR}",PYTHON_BIN="${PYTHON_BIN}",RUN_ID="${RUN_ID}",RUN_DIR="${RUN_DIR}",TASK_STATUS_DIR="${TASK_STATUS_DIR}",TASK_LOG_DIR="${TASK_LOG_DIR}",MERGE_STATUS_FILE="${MERGE_STATUS_FILE}",MERGE_LOG="${MERGE_LOG}",FAILURE_SUMMARY_FILE="${FAILURE_SUMMARY_FILE}",EXPERIMENT_ARGS_FILE="${ARGS_FILE}",ARRAY_JOB_ID="${array_job_id}" \
        "${SCRIPT_DIR}/viewpoint_merge.sbatch"
)"
merge_job_id="${merge_job_id_raw%%;*}"

{
    printf 'array_job_id=%q\n' "${array_job_id}"
    printf 'merge_job_id=%q\n' "${merge_job_id}"
} >> "${SUBMISSION_FILE}"

echo "Submitted array job: ${array_job_id}"
echo "Submitted merge job: ${merge_job_id} (afterany:${array_job_id})"
echo "Output directory: ${OUTPUT_DIR}"
echo "Run directory: ${RUN_DIR}"
echo "Task status directory: ${TASK_STATUS_DIR}"
echo "Merge status file: ${MERGE_STATUS_FILE}"
if (( ${#EXTRA_ARGS[@]} > 0 )); then
    echo "Extra experiment args file: ${ARGS_FILE}"
fi
