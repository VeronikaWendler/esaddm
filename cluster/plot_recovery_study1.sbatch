#!/bin/bash
#SBATCH --job-name=esaddm_rec_plot_s1
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail

# Use the fixed esaddm project root
REPO="${ESADDM_ROOT:-/rds/projects/z/zhanglp-vwendler-core/esaddm}"
test -d "$REPO/studies" || { echo "Missing project root: $REPO" >&2; exit 1; }
cd "$REPO"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/mpl_esaddm_${SLURM_JOB_ID:-local}_study1"
mkdir -p "$MPLCONFIGDIR"

echo "repo=$REPO; study=study1; host=$(hostname)"
PLOT="$REPO/scripts/plot_recovery.py"
test -f "$PLOT" || { echo "Missing $PLOT" >&2; exit 1; }

# Use the current Python env if it has plotting dependencies; otherwise,
# fall back to the existing apptainer image (override IMAGE as necessary).
if python -c 'import numpy, pandas, matplotlib' >/dev/null 2>&1; then
    python "$PLOT" --root "$REPO" --study study1
else
    IMAGE="${IMAGE:-$HOME/containers/hddm_latest.sif}"
    command -v apptainer >/dev/null || { echo 'Need numpy/pandas/matplotlib or apptainer' >&2; exit 1; }
    test -f "$IMAGE" || { echo "Missing container: $IMAGE" >&2; exit 1; }
    apptainer exec --cleanenv --bind "$REPO:$REPO" --env MPLBACKEND=Agg "$IMAGE" \
        python "$PLOT" --root "$REPO" --study study1
fi

echo "Study 1 recovery figures completed in $REPO/studies/study1/results/recovery/esaddm_z/summary"
