#!/usr/bin/env bash
# submit one or more parameter-recovery datasets for a study
# examples: bash cluster/submit_recovery.sh study1 0
#           bash cluster/submit_recovery.sh both all
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$root"
target="${1:?Usage: bash cluster/submit_recovery.sh study1|study2|both 0..9|all}"
rep="${2:?Specify a zero-based recovery repetition (0..9) or all}"
case "$target" in study1|study2|both) ;; *) echo "Invalid study: $target" >&2; exit 2;; esac
reps="${REC_REPS:-10}"
[[ "$reps" =~ ^[0-9]+$ ]] && (( reps >= 1 )) || { echo 'REC_REPS must be a positive integer' >&2; exit 2; }
if [[ "$rep" == all ]]; then
  array="0-$((reps-1))%${REC_MAX_CONCURRENT:-1}"
elif [[ "$rep" =~ ^[0-9]+$ ]] && (( rep < reps )); then
  array="$rep"
else
  echo "Replicate must be 0..$((reps-1)) or all" >&2; exit 2
fi
mkdir -p "$root/logs/slurm"
if [[ "$target" == both ]]; then studies=(study1 study2); else studies=("$target"); fi
for study in "${studies[@]}"; do
  [[ -d "$root/studies/$study/results/fits/esaddm_z" ]] || {
    echo "Missing full ESaDDM+z fits for $study" >&2; exit 3; }
  job=$(sbatch --parsable --chdir="$root" --job-name="esrec_$study" \
    --array="$array" --time="${REC_TIME:-24:00:00}" \
    --mem="${REC_MEM:-180G}" --cpus-per-task="${REC_CPUS:-4}" \
    --export="ALL,REC_STUDY=$study" \
    --output="$root/logs/slurm/esrec_${study}_%A_%a.out" \
    --error="$root/logs/slurm/esrec_${study}_%A_%a.err" \
    "$root/cluster/recovery.sbatch")
  echo "$study parameter-recovery job array: $job, replicate(s): $array"
done
