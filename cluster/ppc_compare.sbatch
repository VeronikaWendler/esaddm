#!/usr/bin/env bash
#SBATCH --job-name=esaddm_ppc_compare
#SBATCH --mail-type=ALL
# All computation is offline: no HDDM fit and no PPC simulation.

set -euo pipefail
module purge
module load bluebear
root="$(pwd -P)"
[[ -f "$root/src/esaddm/ppc_compare.py" ]] || { echo "Missing ppc_compare.py in $root" >&2; exit 1; }
image="${IMAGE:-$HOME/containers/hddm_latest.sif}"
[[ -f "$image" ]] || { echo "Missing Python container: $image" >&2; exit 1; }
study="${1:?study1 or study2 required}"
cache_root="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
cache="$cache_root/esaddm_ppc_compare_${SLURM_JOB_ID:-local}"
mkdir -p "$cache"
flags=()
[[ "${PPC_COMPARE_NO_BANDS:-0}" == 1 ]] && flags+=(--no-bands)
[[ "${PPC_COMPARE_REBUILD_MEDIANS:-0}" == 1 ]] && flags+=(--rebuild-median-cache)
apptainer exec --cleanenv --bind "$root:$root" --pwd "$root" \
  "$image" env PYTHONPATH="$root/src" PYTHONUNBUFFERED=1 MPLBACKEND=Agg \
  MPLCONFIGDIR="$cache" python -m esaddm.ppc_compare --study "$study" "${flags[@]}"
