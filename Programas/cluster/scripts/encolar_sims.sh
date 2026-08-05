#!/usr/bin/env bash
set -euo pipefail

# Use a safe name, don't overwrite PATH
CWD="$(pwd)"
if [ ! -f "config.toml" ]; then
  echo "No se halló config.toml en la dir actual"
  exit 1
fi
# Submit job and capture output
out="$(sbatch slurm_submit_JKR.sh)" || { echo "sbatch failed"; exit 1; }

# Extract job id (e.g., "Submitted batch job 93804")
jobid="$(echo "$out" | awk '{print $4}')"
if [ -z "$jobid" ]; then
  echo "Failed to get job id from sbatch output: $out" >&2
  exit 1
fi

# Ensure destination exists and copy using jobid
mkdir -p "${CWD}/configs"
cp "config.toml" "${CWD}/configs/config_${jobid}.toml"

echo "Copied config to ${CWD}/configs/config_${jobid}.toml"

