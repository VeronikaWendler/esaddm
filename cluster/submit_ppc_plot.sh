#!/usr/bin/env bash
# After PPC generation succeeds, create/recreate tables and plots, e.g.:
#   bash cluster/submit_ppc_plot.sh study1 esaddm_z
#   bash cluster/submit_ppc_plot.sh study1 all
# No HDDM simulation is performed by the plotting stage.
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
    echo "Submitting PPC PLOTS $st/$md" >&2
    sbatch --chdir="$root" --job-name="ppcplot_${st}_${md}" \
      --time="${PPC_PLOT_WALLTIME:-01:00:00}" --mem="${PPC_PLOT_MEM:-8G}" \
      --cpus-per-task=1 \
      --output="$root/logs/slurm/ppcplot_${st}_${md}_%j.out" \
      --error="$root/logs/slurm/ppcplot_${st}_${md}_%j.err" \
      "$root/cluster/ppc_plot.sbatch" "$st" "$md"
  done
done
