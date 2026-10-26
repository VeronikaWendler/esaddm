#!/usr/bin/env bash
# Examples:
#   bash cluster/submit_ppc.sh study1 esaddm_z
#   bash cluster/submit_ppc.sh study1 all
#   PPC_DRAWS=500 bash cluster/submit_ppc.sh study2 all  -> change draws here if you like (we use 500 simialr to the official hddm guidelines)


set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
study="${1:-}"
model="${2:-}"
case "$study" in study1|study2|both) ;; *) echo 'Study: study1|study2|both' >&2; exit 2 ;; esac
case "$model" in accuracy_es|accuracy_ee|addm|addm_z|esaddm|esaddm_z|all) ;;
  *) echo 'Model: accuracy_es|accuracy_ee|addm|addm_z|esaddm|esaddm_z|all' >&2; exit 2 ;;
esac
mkdir -p "$root/logs/slurm"
studies=("$study")
models=("$model")
[[ "$study" == both ]] && studies=(study1 study2)
[[ "$model" == all ]] && models=(accuracy_es accuracy_ee addm addm_z esaddm esaddm_z)
for st in "${studies[@]}"; do
  for md in "${models[@]}"; do
    echo "Submitting PPC $st/$md" >&2
    sbatch --chdir="$root" --job-name="ppc_${st}_${md}" \
      --time="${PPC_WALLTIME:-24:00:00}" --mem="${PPC_MEM:-244G}" \
      --cpus-per-task=5 \
      --output="$root/logs/slurm/ppc_${st}_${md}_%j.out" \
      --error="$root/logs/slurm/ppc_${st}_${md}_%j.err" \
      "$root/cluster/ppc.sbatch" "$st" "$md"
  done
done
