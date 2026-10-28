#!/bin/bash
set -euo pipefail
REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
cd "$REPO"
sbatch cluster/plot_recovery_study1.sbatch
sbatch cluster/plot_recovery_study2.sbatch
