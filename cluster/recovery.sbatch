#!/usr/bin/env bash
#SBATCH --job-name=esrec
#SBATCH --mail-type=ALL
set -euo pipefail
module purge
module load bluebear
root="$(pwd -P)"
[[ -f "$root/src/esaddm/recovery.py" ]] || { echo "Not in esaddm repository: $root" >&2; exit 2; }
image="${IMAGE:-$HOME/containers/hddm_latest.sif}"
[[ -f "$image" ]] || { echo "Missing HDDM image: $image" >&2; exit 2; }
: "${REC_STUDY:?Specify REC_STUDY}"
: "${SLURM_ARRAY_TASK_ID:?Recovery requires an array replicate ID}"

args=(--study "$REC_STUDY" --rep "$SLURM_ARRAY_TASK_ID"
      --reps "${REC_REPS:-10}" --chains "${REC_CHAINS:-5}"
      --samples "${REC_SAMPLES:-5000}" --burn "${REC_BURN:-1000}"
      --jobs "${SLURM_CPUS_PER_TASK:-4}" --seed "${REC_SEED:-20261026}")
[[ "${REC_KEEP_TRACES:-0}" == 1 ]] && args+=(--keep-traces)

apptainer exec --cleanenv --bind "$root:$root" --pwd "$root" "$image" \
  env PYTHONPATH="$root/src" PYTHONUNBUFFERED=1 MPLBACKEND=Agg \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  python -m esaddm.recovery "${args[@]}"
