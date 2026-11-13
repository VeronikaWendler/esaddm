#!/bin/bash
# examples: bash cluster/submit_recovery_plots.sh study1 0
#           bash cluster/submit_recovery_plots.sh both all

set -euo pipefail
ROOT="/rds/projects/z/zhanglp-vwendler-core/esaddm"
[[ -d "$ROOT/logs/slurm" ]] || mkdir -p "$ROOT/logs/slurm"
[[ -f "$ROOT/scripts/plot_recovery.py" ]] || { echo "missing $ROOT/scripts/plot_recovery.py" >&2; exit 1; }
cd "$ROOT"
for study in study1 study2; do
  echo "submitting $study recovery plotting from $ROOT"
  sbatch "cluster/plot_recovery_${study}.sbatch"
done
