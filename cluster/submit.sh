#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
action="${1:-}"
study="${2:-both}"
if [[ "$action" != 'prep' ]]; then
    echo 'Usage: ./cluster/submit.sh prep study1|study2|both' >&2
    exit 2
fi
case "$study" in study1|study2|both) ;; *) echo "Invalid study: $study" >&2; exit 2 ;; esac
mkdir -p "$root/logs/slurm"

# All Slurm output+errors in one file per job
sbatch --chdir="$root" \
    --output="$root/logs/slurm/prep_${study}_%j.log" \
    --error="$root/logs/slurm/prep_${study}_%j.log" \
    "$root/cluster/prep.sbatch" "$study"
