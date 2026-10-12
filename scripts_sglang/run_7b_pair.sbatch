#!/bin/bash
#SBATCH --job-name=bcplus-7b-full-w8
#SBATCH --account=k2m
#SBATCH --partition=main
#SBATCH --qos=k2m
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --constraint=nvidia_h200
#SBATCH --array=0-63%64
#SBATCH --output=logs/slurm-%A_%a.log

set -euo pipefail
if (( $# != 3 )); then
  echo "Usage: sbatch run_7b_pair.sbatch CONFIG OUTPUT_ROOT COMPATIBLE_AGENT_SHA256" >&2
  exit 2
fi
config_path="$1"
output_root="$2"
compatible_agent_sha="$3"
task_id="${SLURM_ARRAY_TASK_ID:?Requires the 64-task comparison array}"
if [[ ! "$task_id" =~ ^[0-9]+$ ]] || (( task_id < 0 || task_id >= 64 )); then
  echo "Array task must be in [0, 64)." >&2
  exit 2
fi
# Interleave full history and W8, with 32 independent replicas of each.
shard_id=$((task_id / 2))
condition=normal
if (( task_id % 2 )); then condition=w8; fi

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export RUSTUP_TOOLCHAIN=1.94.0
repo_dir="${BROWSECOMP_ROOT:-/mnt/weka/home/seungwook.han/BrowseComp-Plus}"
env_prefix="${SGLANG_ENV_PREFIX:-/mnt/weka/home/seungwook.han/miniforge3/envs/sglang-kv-evict}"
cd "$repo_dir"
exec "$env_prefix/bin/python" scripts_sglang/run_local.py agent \
  --config "$config_path" \
  --condition "$condition" \
  --output-dir "$output_root/$condition" \
  --logs-dir "$repo_dir/logs/job-${SLURM_ARRAY_JOB_ID}_${task_id}" \
  -- --limit 0 --max-tokens 32768 --max-iterations 0 --workers 1 \
  --temperature 1.0 --top-p 0.95 --seed 0 \
  --num-shards 32 --shard-id "$shard_id" \
  --resume-compatible-agent-sha256 "$compatible_agent_sha"
