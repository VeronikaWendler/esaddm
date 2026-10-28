#!/bin/bash
#SBATCH --job-name=esrecplot_study1
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=/rds/projects/z/zhanglp-vwendler-core/esaddm/logs/slurm/esrecplot_study1_%j.out
#SBATCH --error=/rds/projects/z/zhanglp-vwendler-core/esaddm/logs/slurm/esrecplot_study1_%j.err
#SBATCH --chdir=/rds/projects/z/zhanglp-vwendler-core/esaddm
set -euo pipefail

# Plot only: no model fitting, posterior sampling, or changes to rep_XX inputs.
REPO="/rds/projects/z/zhanglp-vwendler-core/esaddm"
STUDY="study1"
PLOT="$REPO/scripts/plot_recovery.py"
[[ -f "$PLOT" ]] || { echo "missing plotter: $PLOT" >&2; exit 1; }
[[ -d "$REPO/logs/slurm" ]] || { echo "missing log directory: $REPO/logs/slurm" >&2; exit 1; }
cd "$REPO"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/mpl_esrecplot_${SLURM_JOB_ID:-local}"
mkdir -p "$MPLCONFIGDIR"

echo "plot job=${SLURM_JOB_ID:-local} study=$STUDY host=$(hostname) root=$REPO"
if python -c 'import numpy, pandas, matplotlib' >/dev/null 2>&1; then
    python -u "$PLOT" --root "$REPO" --study "$STUDY"
else
    # Same container fallback used by the previous recovery plotting batch files.
    IMAGE="${ESADDM_PLOT_IMAGE:-$HOME/containers/hddm_latest.sif}"
    command -v apptainer >/dev/null 2>&1 || { echo 'python plotting packages missing; apptainer unavailable' >&2; exit 1; }
    [[ -f "$IMAGE" ]] || { echo "missing plotting image: $IMAGE" >&2; exit 1; }
    apptainer exec --cleanenv --bind "$REPO:$REPO" --env MPLBACKEND=Agg "$IMAGE" \
        python -u "$PLOT" --root "$REPO" --study "$STUDY"
fi

echo "success: $REPO/studies/$STUDY/results/recovery/esaddm_z/summary"
