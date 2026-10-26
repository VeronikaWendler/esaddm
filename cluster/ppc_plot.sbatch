#!/usr/bin/env bash
#SBATCH --job-name=esaddm_ppc_plot
#SBATCH --mail-type=ALL
set -euo pipefail
module purge
module load bluebear
root="$(pwd -P)"
[[ -f "$root/src/esaddm/ppc_plot.py" ]] || { echo "Invalid root: $root" >&2; exit 1; }
image="${IMAGE:-$HOME/containers/hddm_latest.sif}"
[[ -f "$image" ]] || { echo "Missing Python image: $image" >&2; exit 1; }
study="${1:?study required}"
model="${2:?model required}"
cache_root="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
mkdir -p "$cache_root/esaddm_ppc_plots_${SLURM_JOB_ID:-local}"
apptainer exec --cleanenv --bind "$root:$root" --pwd "$root" \
  "$image" env PYTHONPATH="$root/src" PYTHONUNBUFFERED=1 MPLBACKEND=Agg \
  MPLCONFIGDIR="$cache_root/esaddm_ppc_plots_${SLURM_JOB_ID:-local}" \
  python -m esaddm.ppc_plot --study "$study" --model "$model"
