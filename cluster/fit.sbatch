#!/usr/bin/env bash
#SBATCH --job-name=esaddm_fit
#SBATCH --mail-type=ALL

set -euo pipefail

module purge
module load bluebear

# submit_fit.sh passes --chdir, so PWD is the repository root.
root="$(pwd -P)"
[[ -f "$root/src/esaddm/fit.py" ]] || {
    echo "Invalid repository root: $root" >&2
    exit 1
}

image="${IMAGE:-$HOME/containers/hddm_latest.sif}"
[[ -f "$image" ]] || {
    echo "Missing HDDM container: $image" >&2
    exit 1
}

action="${1:?action required}"
study="${2:?study required}"
model="${3:?model required}"

args=("$action" --study "$study" --model "$model")

if [[ "$action" == run ]]; then
    args+=(--chains 5 --workers 5 --samples "${SAMPLES:-5000}"
           --burn "${BURN:-1000}" --seed "${SEED:-20260919}")
elif [[ "$action" == smoke ]]; then
    args+=(--chains 2 --workers 2 --samples "${SMOKE_SAMPLES:-150}"
           --burn "${SMOKE_BURN:-50}" --seed "${SEED:-20260919}")
fi

apptainer exec --cleanenv --bind "$root:$root" --pwd "$root" \
    "$image" env PYTHONPATH="$root/src" \
    python -m esaddm.fit "${args[@]}"
