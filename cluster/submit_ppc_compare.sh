#!/usr/bin/env bash
# Usage: bash cluster/submit_ppc_compare.sh study1|study2|both
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
case "${1:-}" in
  study1|study2) studies=("$1");;
  both) studies=(study1 study2);;
  *) echo 'Usage: bash cluster/submit_ppc_compare.sh study1|study2|both' >&2; exit 2;;
esac
mkdir -p "$root/logs/slurm"
for study in "${studies[@]}"; do
  echo "Submitting four-model PPC comparison for $study (offline, no simulations)" >&2
  sbatch --chdir="$root" --job-name="ppccompare_${study}" \
    --cpus-per-task=1 \
    --time="${PPC_COMPARE_WALLTIME:-02:00:00}" \
    --mem="${PPC_COMPARE_MEM:-12G}" \
    --output="$root/logs/slurm/ppccompare_${study}_%j.out" \
    --error="$root/logs/slurm/ppccompare_${study}_%j.err" \
    "$root/cluster/ppc_compare.sbatch" "$study"
done
