#!/usr/bin/env bash
#SBATCH --job-name=viewpoint-merge
#SBATCH --output=slurm/logs/viewpoint_merge_%j.out
#SBATCH --error=slurm/logs/viewpoint_merge_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=04:00:00

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "${PROJECT_ROOT}"

NUM_SHARDS="${NUM_SHARDS:-160}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs_viewpoint_pymdp_study1_array}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ID="${RUN_ID:-manual}"
RUN_DIR="${RUN_DIR:-${OUTPUT_DIR}/run_logs/${RUN_ID}}"
TASK_STATUS_DIR="${TASK_STATUS_DIR:-${RUN_DIR}/tasks}"
TASK_LOG_DIR="${TASK_LOG_DIR:-${RUN_DIR}/task_logs}"
MERGE_STATUS_FILE="${MERGE_STATUS_FILE:-${RUN_DIR}/merge.status}"
MERGE_LOG="${MERGE_LOG:-${RUN_DIR}/merge.log}"
FAILURE_SUMMARY_FILE="${FAILURE_SUMMARY_FILE:-${RUN_DIR}/failure_summary.txt}"
EXPERIMENT_ARGS_FILE="${EXPERIMENT_ARGS_FILE:-}"
ARRAY_JOB_ID="${ARRAY_JOB_ID:-}"

timestamp() {
    date -u +%Y-%m-%dT%H:%M:%SZ
}

STARTED_AT="$(timestamp)"
LAST_FAILED_COMMAND=""
LAST_FAILED_LINE=""

write_status() {
    local state="$1"
    local exit_code="$2"
    local updated_at finished_at
    updated_at="$(timestamp)"
    finished_at=""
    if [[ "${state}" != "running" ]]; then
        finished_at="${updated_at}"
    fi
    {
        printf 'state=%q\n' "${state}"
        printf 'run_id=%q\n' "${RUN_ID}"
        printf 'started_at=%q\n' "${STARTED_AT}"
        printf 'updated_at=%q\n' "${updated_at}"
        printf 'finished_at=%q\n' "${finished_at}"
        printf 'exit_code=%q\n' "${exit_code}"
        printf 'project_root=%q\n' "${PROJECT_ROOT}"
        printf 'output_dir=%q\n' "${OUTPUT_DIR}"
        printf 'job_id=%q\n' "${SLURM_JOB_ID:-}"
        printf 'array_job_id=%q\n' "${ARRAY_JOB_ID}"
        printf 'hostname=%q\n' "$(hostname)"
        printf 'merge_log=%q\n' "${MERGE_LOG}"
        printf 'failure_summary_file=%q\n' "${FAILURE_SUMMARY_FILE}"
        printf 'last_failed_line=%q\n' "${LAST_FAILED_LINE}"
        printf 'last_failed_command=%q\n' "${LAST_FAILED_COMMAND}"
    } > "${MERGE_STATUS_FILE}"
}

record_failure_summary() {
    local exit_code="$1"
    {
        echo
        echo "Merge step failure"
        echo "generated_at=$(timestamp)"
        echo "merge_job_id=${SLURM_JOB_ID:-}"
        echo "merge_exit_code=${exit_code}"
        echo "merge_log=${MERGE_LOG}"
        if [[ -n "${LAST_FAILED_LINE}" ]]; then
            echo "last_failed_line=${LAST_FAILED_LINE}"
        fi
        if [[ -n "${LAST_FAILED_COMMAND}" ]]; then
            echo "last_failed_command=${LAST_FAILED_COMMAND}"
        fi
    } >> "${FAILURE_SUMMARY_FILE}"
}

on_exit() {
    local exit_code=$?
    if (( exit_code == 0 )); then
        echo "[$(date)] Merge step completed successfully."
        write_status success 0
    else
        echo "[$(date)] Merge step failed with exit code ${exit_code}."
        record_failure_summary "${exit_code}"
        write_status failed "${exit_code}"
    fi
}

collect_shard_failures() {
    local shard_index status_path failures task_tag
    failures=0
    {
        echo "Run failure summary"
        echo "generated_at=$(timestamp)"
        echo "run_id=${RUN_ID}"
        echo "output_dir=${OUTPUT_DIR}"
        echo "array_job_id=${ARRAY_JOB_ID}"
        echo "merge_job_id=${SLURM_JOB_ID:-}"
        echo "task_status_dir=${TASK_STATUS_DIR}"
        echo
    } > "${FAILURE_SUMMARY_FILE}"

    for (( shard_index = 0; shard_index < NUM_SHARDS; shard_index++ )); do
        task_tag="$(printf 'task_%04d' "${shard_index}")"
        status_path="${TASK_STATUS_DIR}/${task_tag}.status"
        if [[ ! -f "${status_path}" ]]; then
            echo "${task_tag}: missing status file (${status_path})" >> "${FAILURE_SUMMARY_FILE}"
            failures=$((failures + 1))
            continue
        fi

        unset state exit_code hostname task_log updated_at finished_at last_failed_line last_failed_command
        # shellcheck disable=SC1090
        source "${status_path}"
        case "${state:-unknown}" in
            success)
                ;;
            running)
                echo "${task_tag}: still marked running; updated_at=${updated_at:-unknown}; task_log=${task_log:-unknown}" >> "${FAILURE_SUMMARY_FILE}"
                failures=$((failures + 1))
                ;;
            failed)
                echo "${task_tag}: failed; exit_code=${exit_code:-unknown}; host=${hostname:-unknown}; task_log=${task_log:-unknown}; line=${last_failed_line:-unknown}; command=${last_failed_command:-unknown}" >> "${FAILURE_SUMMARY_FILE}"
                failures=$((failures + 1))
                ;;
            *)
                echo "${task_tag}: unexpected state=${state:-unknown}; task_log=${task_log:-unknown}" >> "${FAILURE_SUMMARY_FILE}"
                failures=$((failures + 1))
                ;;
        esac
    done

    if (( failures > 0 )); then
        echo "Detected ${failures} failed or incomplete shard(s). Merge will not run."
        return 1
    fi

    rm -f "${FAILURE_SUMMARY_FILE}"
}

trap 'LAST_FAILED_COMMAND=$BASH_COMMAND; LAST_FAILED_LINE=$LINENO' ERR
trap on_exit EXIT

mkdir -p "${OUTPUT_DIR}" "${RUN_DIR}" "${TASK_STATUS_DIR}" "${TASK_LOG_DIR}"
write_status running 0
exec > >(tee -a "${MERGE_LOG}") 2>&1

EXTRA_ARGS=()
if [[ -n "${EXPERIMENT_ARGS_FILE}" ]]; then
    if [[ ! -f "${EXPERIMENT_ARGS_FILE}" ]]; then
        echo "Experiment args file is missing: ${EXPERIMENT_ARGS_FILE}"
        exit 2
    fi
    mapfile -t EXTRA_ARGS < "${EXPERIMENT_ARGS_FILE}"
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export JAX_PLATFORM_NAME=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false

collect_shard_failures

COMMAND=(
    "${PYTHON_BIN}" "${PROJECT_ROOT}/viewpoint_inference_experiment.py"
    --output-dir "${OUTPUT_DIR}"
    --num-shards "${NUM_SHARDS}"
    --merge-shards
    --no-progress
    "${EXTRA_ARGS[@]}"
)

echo "[$(date)] Merging ${NUM_SHARDS} shard(s) from ${OUTPUT_DIR}"
echo "[$(date)] Merge status file: ${MERGE_STATUS_FILE}"
echo "[$(date)] Merge log: ${MERGE_LOG}"
printf '[%s] Command:' "$(date)"
printf ' %q' "${COMMAND[@]}"
printf '\n'
"${COMMAND[@]}"
echo "[$(date)] Merge finished"
