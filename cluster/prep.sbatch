#!/usr/bin/env bash
#SBATCH --job-name=esaddm_prep
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --mail-type=ALL

set -euo pipefail

module purge
module load bluebear

root="$(pwd -P)"
study="${1:?usage: prep.sbatch study1|study2|both}"
image="${IMAGE:-${HOME}/containers/hddm_latest.sif}"
if [[ ! -f "$image" ]]; then echo "Missing container: $image" >&2; exit 1; fi

run_study() {
    local this_study="$1" prepared_dir="$root/studies/$1/data/prepared"
    local count=0
    local name
    for name in identity_es accuracy_es accuracy_ee; do
        if [[ -f "$prepared_dir/$name.csv" ]]; then ((count+=1)); fi
    done
    local action="prepare"
    if (( count == 3 )); then
        action="verify"
    elif (( count != 0 )); then
        echo "Incomplete prepared inputs ($count/3) for $this_study; refusing to overwrite." >&2
        exit 1
    fi
    echo "${this_study}: running ${action} in the existing HDDM container"
    # Both studies and output folders are in the same repository

    apptainer exec --cleanenv \
        --bind "${root}:${root}" \
        --pwd "$root" \
        "$image" \
        env PYTHONPATH="$root/src" \
        python -m esaddm.cli "$action" --study "$this_study"
}

case "$study" in
    study1|study2) run_study "$study" ;;
    both) run_study study1; run_study study2 ;;
    *) echo "Invalid study '$study'" >&2; exit 2 ;;
esac
