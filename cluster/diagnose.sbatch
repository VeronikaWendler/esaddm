#!/usr/bin/env bash
#SBATCH --job-name=esaddm_diagnostics
#SBATCH --mail-type=ALL
set -euo pipefail
module purge
module load bluebear
root="$(pwd -P)"
[[ -f "$root/src/esaddm/diagnose.py" ]] || { echo "Incorrect working directory: $root" >&2; exit 1; }
image="${IMAGE:-$HOME/containers/hddm_latest.sif}"
[[ -f "$image" ]] || { echo "HDDM container not found: $image" >&2; exit 1; }
study="${1:?study required}"
model="${2:?model required}"
stage="${3:-fits}"
chains="${CHAINS:-5}"
[[ "$stage" == smoke ]] && chains="${CHAINS:-2}"
args=(--study "$study" --model "$model" --stage "$stage" --chains "$chains")
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/esaddm_mpl_${SLURM_JOB_ID:-manual}"
mkdir -p "$MPLCONFIGDIR"
apptainer exec --cleanenv --bind "$root:$root" --pwd "$root" \
  --bind "$MPLCONFIGDIR:$MPLCONFIGDIR" \
  "$image" env PYTHONPATH="$root/src" MPLBACKEND=Agg MPLCONFIGDIR="$MPLCONFIGDIR" \
  python -m esaddm.diagnose "${args[@]}"
