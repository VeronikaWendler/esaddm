#!/usr/bin/env bash
# This is how you submit the models:
#   bash cluster/submit_fit.sh smoke study1 esaddm_z
#   bash cluster/submit_fit.sh run study1 esaddm_z
#   bash cluster/submit_fit.sh run both all
#   bash cluster/submit_fit.sh run study1 all
#   bash cluster/submit_fit.sh smoke study1 all

set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
action="${1:-}"
study="${2:-}"
model="${3:-}"
case "$action" in check|smoke|run) ;; *) echo 'Use: submit_fit.sh check|smoke|run STUDY MODEL' >&2; exit 2 ;; esac
case "$study" in study1|study2|both) ;; *) echo 'Study: study1, study2 or both' >&2; exit 2 ;; esac
case "$model" in accuracy_es|accuracy_ee|addm|addm_z|esaddm|esaddm_z|all) ;;
    *) echo 'Unknown model' >&2; exit 2 ;;
esac

if [[ "$action" == smoke && "$study" == both ]]; then
    echo 'Smoke testing requires one study' >&2
    exit 2
fi

mkdir -p "$root/logs/slurm"

submit_one() {
    local st="$1" md="$2" stem="$action" time memory cpus
    case "$action" in
        check) time=00:30:00; memory=8G; cpus=1 ;;
        smoke) time=04:00:00; memory=32G; cpus=2 ;;
        run) time=24:00:00; memory=180G; cpus=6 ;;
    esac
    echo "Submitting $action $st/$md" >&2
    sbatch --chdir="$root" --time="$time" --cpus-per-task="$cpus" --mem="$memory" \
        --job-name="esaddm_${st}_${md}" \
        --output="$root/logs/slurm/${stem}_${st}_${md}_%j.out" \
        --error="$root/logs/slurm/${stem}_${st}_${md}_%j.err" \
        "$root/cluster/fit.sbatch" "$action" "$st" "$md"
}

if [[ "$action" == check ]]; then
    # single diagnostic job, even when checking both studies and all models
    submit_one "$study" "$model"
elif [[ "$action" == smoke ]]; then
    submit_one "$study" "$model"
else
    studies=("$study")
    models=("$model")
    [[ "$study" == both ]] && studies=(study1 study2)
    [[ "$model" == all ]] && models=(accuracy_es accuracy_ee addm addm_z esaddm esaddm_z)
    # One job per study and model
    for st in "${studies[@]}"; do
        for md in "${models[@]}"; do
            submit_one "$st" "$md"
        done
    done
fi
