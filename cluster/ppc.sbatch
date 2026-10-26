#!/usr/bin/env bash
#SBATCH --job-name=esaddm_ppc
#SBATCH --mail-type=ALL
set -euo pipefail
module purge
module load bluebear

# submit_ppc.sh uses sbatch --chdir on the actual project directory.
root="$(pwd -P)"
[[ -f "$root/src/esaddm/ppc.py" ]] || { echo "Invalid root: $root" >&2; exit 1; }
image="${IMAGE:-$HOME/containers/hddm_latest.sif}"
[[ -f "$image" ]] || { echo "Missing HDDM image: $image" >&2; exit 1; }
study="${1:?study required}"
model="${2:?model required}"

# Match the fitting container; keep Matplotlib's config in a writable location.
cache_root="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
mkdir -p "$cache_root/esaddm_mpl_${SLURM_JOB_ID:-local}"

apptainer exec --cleanenv --bind "$root:$root" --pwd "$root" \
  "$image" env PYTHONPATH="$root/src" \
  PYTHONUNBUFFERED=1 MPLBACKEND=Agg \
  MPLCONFIGDIR="$cache_root/esaddm_mpl_${SLURM_JOB_ID:-local}" \
  python -m esaddm.ppc --study "$study" --model "$model" \
    --chains "${CHAINS:-5}" --draws "${PPC_DRAWS:-500}" --batch-size "${PPC_BATCH_SIZE:-20}" \
    --seed "${PPC_SEED:-20260919}"
