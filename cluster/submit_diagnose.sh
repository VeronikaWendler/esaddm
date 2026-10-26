#!/usr/bin/env bash
# One Slurm job and log set per submission, all models processed sequentially.
# Examples:
#   bash cluster/submit_diagnose.sh study1 all
#   bash cluster/submit_diagnose.sh study1 esaddm_z
#

set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
study="${1:-}"
model="${2:-}"
stage="${3:-fits}"
case "$study" in study1|study2) ;; *) echo 'Study must be study1 or study2' >&2; exit 2 ;; esac
case "$model" in accuracy_es|accuracy_ee|addm|addm_z|esaddm|esaddm_z|all) ;; *) echo 'Unknown model' >&2; exit 2 ;; esac
case "$stage" in fits|smoke) ;; *) echo 'Stage must be fits or smoke' >&2; exit 2 ;; esac
mkdir -p "$root/logs/slurm"
sbatch --chdir="$root" --time="${DIAG_TIME:-12:00:00}" \
    --cpus-per-task=1 --mem="${DIAG_MEM:-64G}" \
    --export=ALL \
    --job-name="diagnose_${study}_${model}" \
    --output="$root/logs/slurm/diagnose_${stage}_${study}_${model}_%j.out" \
    --error="$root/logs/slurm/diagnose_${stage}_${study}_${model}_%j.err" \
    "$root/cluster/diagnose.sbatch" "$study" "$model" "$stage"
