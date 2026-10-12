#!/bin/bash
#SBATCH --job-name=browsecomp-horizon
#SBATCH --account=k2m
#SBATCH --partition=main
#SBATCH --qos=k2m
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --constraint=nvidia_h200
#SBATCH --output=logs/slurm-%j.log

set -euo pipefail
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export RUSTUP_TOOLCHAIN=1.94.0
repo_dir="${BROWSECOMP_ROOT:-/mnt/weka/home/seungwook.han/BrowseComp-Plus}"
env_prefix="${SGLANG_ENV_PREFIX:-/mnt/weka/home/seungwook.han/miniforge3/envs/sglang-kv-evict}"
cd "$repo_dir"
mode="${1:-smoke}"
if (( $# )); then shift; fi
log_id="${SLURM_JOB_ID}"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  if [[ "$mode" != "agent" ]]; then
    echo "Array tasks must use agent mode; submit grading after the whole array." >&2
    exit 2
  fi
  # The submitter supplies --num-shards explicitly, so requeuing or retrying
  # an individual array index keeps the original question partition.
  set -- "$@" --shard-id "$SLURM_ARRAY_TASK_ID"
  log_id="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
fi
exec "$env_prefix/bin/python" scripts_sglang/run_local.py "$mode" \
  --logs-dir "$repo_dir/logs/job-${log_id}" "$@"
