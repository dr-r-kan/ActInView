#!/usr/bin/env bash
#SBATCH --job-name=viewpoint-ai
#SBATCH --output=slurm/logs/viewpoint_%A_%a.out
#SBATCH --error=slurm/logs/viewpoint_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
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
EXPERIMENT_ARGS_FILE="${EXPERIMENT_ARGS_FILE:-}"

timestamp() {
    date -u +%Y-%m-%dT%H:%M:%SZ
}

STARTED_AT="$(timestamp)"
LAST_FAILED_COMMAND=""
LAST_FAILED_LINE=""
TASK_TAG="task_unknown"
TASK_LOG=""
STATUS_FILE=""

write_status() {
    local state="$1"
    local exit_code="$2"
    if [[ -z "${STATUS_FILE}" ]]; then
        return 0
    fi
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
        printf 'array_job_id=%q\n' "${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-}}"
        printf 'array_task_id=%q\n' "${SLURM_ARRAY_TASK_ID:-}"
        printf 'task_tag=%q\n' "${TASK_TAG}"
        printf 'hostname=%q\n' "$(hostname)"
        printf 'task_log=%q\n' "${TASK_LOG}"
        printf 'status_file=%q\n' "${STATUS_FILE}"
        printf 'last_failed_line=%q\n' "${LAST_FAILED_LINE}"
        printf 'last_failed_command=%q\n' "${LAST_FAILED_COMMAND}"
    } > "${STATUS_FILE}"
}

on_exit() {
    local exit_code=$?
    if (( exit_code == 0 )); then
        echo "[$(date)] Completed ${TASK_TAG} successfully."
        write_status success 0
    else
        echo "[$(date)] ${TASK_TAG} failed with exit code ${exit_code}."
        write_status failed "${exit_code}"
    fi
}

trap 'LAST_FAILED_COMMAND=$BASH_COMMAND; LAST_FAILED_LINE=$LINENO' ERR
trap on_exit EXIT

mkdir -p "${OUTPUT_DIR}" "${TASK_STATUS_DIR}" "${TASK_LOG_DIR}"

if [[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-9]+$ ]]; then
    TASK_TAG="$(printf 'task_%04d' "${SLURM_ARRAY_TASK_ID}")"
else
    TASK_TAG="task_invalid"
fi
TASK_LOG="${TASK_LOG_DIR}/${TASK_TAG}.log"
STATUS_FILE="${TASK_STATUS_DIR}/${TASK_TAG}.status"
write_status running 0

exec > >(tee -a "${TASK_LOG}") 2>&1

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "SLURM_ARRAY_TASK_ID is not set. Submit this script with sbatch --array."
    exit 2
fi

if (( SLURM_ARRAY_TASK_ID >= NUM_SHARDS )); then
    echo "Array task ${SLURM_ARRAY_TASK_ID} exceeds NUM_SHARDS=${NUM_SHARDS}."
    exit 2
fi

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

COMMAND=(
    "${PYTHON_BIN}" "${PROJECT_ROOT}/viewpoint_inference_experiment.py"
    --output-dir "${OUTPUT_DIR}"
    --num-shards "${NUM_SHARDS}"
    --shard-index "${SLURM_ARRAY_TASK_ID}"
    --no-progress
    "${EXTRA_ARGS[@]}"
)

echo "[$(date)] Starting ${TASK_TAG} (${SLURM_ARRAY_TASK_ID}/${NUM_SHARDS}) -> ${OUTPUT_DIR}"
echo "[$(date)] Status file: ${STATUS_FILE}"
echo "[$(date)] Task log: ${TASK_LOG}"
printf '[%s] Command:' "$(date)"
printf ' %q' "${COMMAND[@]}"
printf '\n'
"${COMMAND[@]}"
echo "[$(date)] Finished ${TASK_TAG}"
