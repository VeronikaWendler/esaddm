"""Prepare the ES/EE model inputs for study 1 and 2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .prepare import prepare_accuracy, prepare_identity_es

ROOT = Path(__file__).resolve().parents[2]
TARGETS = ("identity_es", "accuracy_es", "accuracy_ee")


def make_inputs(raw: pd.DataFrame, config: dict) -> dict[str, pd.DataFrame]:
    excluded = config["excluded_subjects"]
    cutoff = config["min_rt_seconds"]
    prepared = {}
    for target in TARGETS:
        if target == "identity_es":
            data, _ = prepare_identity_es(raw, excluded, cutoff)
        else:
            data, _ = prepare_accuracy(raw, target[-2:].upper(), excluded, cutoff)
        expected = config["expected_rows"][target]
        if len(data) != expected or data.subj_idx.nunique() != config["expected_participants"]:
            raise ValueError(f"{target}: unexpected number of trials or participants")
        prepared[target] = data
    return prepared


def prepare_study(study: str, action: str, force: bool = False) -> None:
    config = json.loads((ROOT / "config" / f"{study}.json").read_text())
    raw_path = ROOT / "studies" / study / "data" / "raw" / config["raw_file"]
    prepared_dir = ROOT / "studies" / study / "data" / "prepared"
    if not raw_path.is_file():
        raise FileNotFoundError(f"Missing dataset: {raw_path}")

    data = make_inputs(pd.read_csv(raw_path, low_memory=False), config)
    existing = [prepared_dir / f"{name}.csv" for name in TARGETS]
    if action == "verify":
        for name, path in zip(TARGETS, existing):
            if not path.is_file():
                raise FileNotFoundError(f"Cannot verify missing model input: {path}")
            expected_csv = data[name].to_csv(index=False, float_format="%.3f").encode("utf-8")
            if path.read_bytes() != expected_csv:
                raise ValueError(f"{path} does not match newly prepared model input")
            print(f"PASS {study}/{name}: {len(data[name])} trials, "
                  f"{data[name].subj_idx.nunique()} participants", flush=True)
        return

    if action != "prepare":
        raise ValueError(f"Unknown action: {action}")
    if any(path.exists() for path in existing) and not force:
        raise FileExistsError("Prepared inputs already exist; use verify or --force")
    prepared_dir.mkdir(parents=True, exist_ok=True)
    for name, path in zip(TARGETS, existing):
        data[name].to_csv(path, index=False, float_format="%.3f")
        print(f"PREPARED {study}/{name}: {len(data[name])} trials", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and verify actual model inputs")
    parser.add_argument("action", choices=("prepare", "verify"))
    parser.add_argument("--study", choices=("study1", "study2", "both"), default="both")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for study in (("study1", "study2") if args.study == "both" else (args.study,)):
        prepare_study(study, args.action, force=args.force)


if __name__ == "__main__":
    main()
