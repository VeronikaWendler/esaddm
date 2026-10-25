
"""Fit the HDDM models for Study 1 and Study 2.

Smoke tests use prepared data and HDDM, save and reload one short
chain per model, and execute model families sequentially in one job.
Full fits use five independently seeded chains per model by default.

"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import numpy as np

from .models import ROOT, build_hddm_model, check_input, load_models



def run_chain(*, study: str, model_name: str, chain: int, samples: int,
              burn: int, base_seed: int, root: Path = ROOT,
              output_type: str = "fits", reload_saved: bool = False) -> Path:
    """Fit one HDDM chain and save HDDM, dill PKL, ArviZ NC, and MCMC DB."""

    import hddm
    import dill
    import arviz as az
    import pandas as pd

    if output_type not in {"fits", "smoke"}:
        raise ValueError("Invalid output type")
    if chain < 1 or not 0 <= burn < samples:
        raise ValueError("Require chain >= 1 and 0 <= burn < samples")

    spec = load_models(root)[model_name]
    data, _ = check_input(study, spec, root)

    out_dir = root / "studies" / study / "results" / output_type / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Globally identifiable filenames
    prefix = f"{model_name}_{study}"
    if output_type == "smoke":
        prefix += "_smoke"

    stem = f"{prefix}_chain{chain}"

    model_path = out_dir / f"{stem}.hddm"
    pkl_path = out_dir / f"{stem}.pkl"
    nc_path = out_dir / f"{stem}.nc"
    db_path = out_dir / f"{stem}_db"

    # Never silently overwrite results
    existing = [
        path for path in (model_path, pkl_path, nc_path)
        if path.exists()
    ]
    existing += list(out_dir.glob(f"{stem}_db*"))

    if existing:
        raise FileExistsError(
            f"Existing outputs for {stem}: {existing}"
        )

    seed = base_seed + chain
    np.random.seed(seed)

    print(
        f"START {stem}: samples={samples}, burn={burn}, seed={seed}",
        flush=True,
    )
    print(f"FORMULA: {spec.drift_formula}", flush=True)
    print(f"INCLUDE: {spec.include}", flush=True)

    model = build_hddm_model(data, spec, hddm_module=hddm)
    model.find_starting_values()

    # Small PPC for smoke, larger PPC for full fits.
    # These numbers can be increased independently of MCMC samples.
    n_ppc = 10 if output_type == "smoke" else 100

    infdata = model.sample(
        samples,
        burn=burn,
        dbname=str(db_path),
        db="pickle",
        return_infdata=True,
        loglike=True,
        ppc=True,
        n_ppc=n_ppc,
    )

    if infdata is None or not hasattr(infdata, "posterior"):
        raise RuntimeError("HDDM did not return valid ArviZ InferenceData")

    # Save native HDDM model
    model.save(str(model_path))

    # Save separate dill-serialized Python model
    with pkl_path.open("wb") as handle:
        dill.dump(model, handle)

    # Handle missing values stored with object dtype before NetCDF
    for group_name in infdata.groups():
        dataset = infdata[group_name]
        for variable in dataset.data_vars:
            values = dataset[variable].values
            if isinstance(values, np.ndarray) and values.dtype == object:
                missing = pd.isna(values)
                if np.any(missing):
                    cleaned = values.copy()
                    cleaned[missing] = np.nan
                    dataset[variable].values = cleaned

    # Save ArviZ InferenceData
    az.to_netcdf(infdata, str(nc_path))

    # Verify that every requested output was created
    required = (model_path, pkl_path, nc_path)
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty output: {path}")

    database_files = list(out_dir.glob(f"{stem}_db*"))
    if not database_files:
        raise RuntimeError(f"MCMC database was not saved for {stem}")

    # Smoke mode: verify files can actually be read again
    if reload_saved:
        restored_hddm = hddm.load(str(model_path))

        with pkl_path.open("rb") as handle:
            restored_pkl = dill.load(handle)

        restored_nc = az.from_netcdf(str(nc_path))

        if restored_hddm is None or restored_pkl is None:
            raise RuntimeError(f"Model reload failed for {stem}")
        if not hasattr(restored_nc, "posterior"):
            raise RuntimeError(f"NetCDF reload failed for {stem}")

        del restored_hddm, restored_pkl, restored_nc

    print(
        f"PASS {stem}: HDDM, PKL, NC and MCMC database saved",
        flush=True,
    )
    return model_path


def run_full(*, study: str, model_name: str, chain_ids: list[int],
             workers: int, samples: int, burn: int, seed: int,
             output_type: str = "fits",
             reload_saved: bool = False) -> None:
    """Run independent chains in one Slurm job, with one run manifest."""
    if not chain_ids or workers < 1:
        raise ValueError("At least one chain and one worker are required")

    spec = load_models()[model_name]
    _, audit = check_input(study, spec)
    out_dir = ROOT / "studies" / study / "results" / output_type / model_name

    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Fit folder is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    record = {
        "study": study,
        "model": model_name,
        "formula": spec.drift_formula,
        "coordinate": spec.coordinate,
        "included_ddm_parameters": spec.include,
        "participants": audit["participants"],
        "trials": audit["rows"],
        "chains": chain_ids,
        "seeds": [seed + chain for chain in chain_ids],
        "samples_per_chain": samples,
        "burn_per_chain": burn,
        "retained_per_chain": samples - burn,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
    }


    prefix = f"{model_name}_{study}"
    if output_type == "smoke":
        prefix += "_smoke"
    run_file = out_dir / f"{prefix}_run.json"

    run_file.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    arguments = dict(
        study=study,
        model_name=model_name,
        samples=samples,
        burn=burn,
        base_seed=seed,
        output_type=output_type,
        reload_saved=reload_saved,
    )

    if workers == 1 or len(chain_ids) == 1:
        for chain in chain_ids:
            run_chain(chain=chain, **arguments)
    else:
        from joblib import Parallel, delayed

        Parallel(n_jobs=min(workers, len(chain_ids)), backend="loky")(
            delayed(run_chain)(chain=chain, **arguments)
            for chain in chain_ids
        )

    record["status"] = "completed"
    record["finished_utc"] = datetime.now(timezone.utc).isoformat()
    run_file.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    print(f"PASS {study}/{model_name}: {len(chain_ids)} chains completed",
          flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="ESaDDM fitting with real HDDM")
    parser.add_argument("action", choices=("check", "smoke", "run"))
    parser.add_argument("--study", choices=("study1", "study2", "both"),
                        required=True)
    parser.add_argument("--model", choices=(*load_models(), "all"),
                        required=True)
    parser.add_argument("--chain", type=int, default=None)
    parser.add_argument("--chains", type=int, default=5)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--burn", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260919)

    args = parser.parse_args(argv)

    if args.action == "check":
        studies = ("study1", "study2") if args.study == "both" else (args.study,)
        names = load_models() if args.model == "all" else (args.model,)

        for study in studies:
            for name in names:
                spec = load_models()[name]
                _, info = check_input(study, spec)
                print(f"PASS check {study}/{name}: {info['rows']} trials, "
                      f"rank {info['design_rank']}/{info['n_predictors']}",
                      flush=True)

        print("PASS: all design checks completed (no MCMC)", flush=True)
        return

    if args.study == "both":
        parser.error("Fitting requires one study at a time")

    samples = (
        args.samples if args.samples is not None
        else (150 if args.action == "smoke" else 5000)
    )
    burn = (
        args.burn if args.burn is not None
        else (50 if args.action == "smoke" else 1000)
    )

    if not 0 <= burn < samples:
        parser.error("Require 0 <= burn < samples")


    if args.action == "smoke":
        if args.chain is not None:
            parser.error("Use --chains for smoke tests")

        if args.chains < 1 or args.workers < 1:
            parser.error("Chains and workers must be positive")

        names = list(load_models()) if args.model == "all" else [args.model]

        for index, name in enumerate(names, start=1):
            print(
                f"SMOKE {index}/{len(names)}: {args.study}/{name}",
                flush=True,
            )

            run_full(
                study=args.study,
                model_name=name,
                chain_ids=list(range(1, args.chains + 1)),
                workers=args.workers,
                samples=samples,
                burn=burn,
                seed=args.seed,
                output_type="smoke",
                reload_saved=True,
            )

            print(f"PASS smoke {args.study}/{name}", flush=True)

        print(
            f"PASS: all {len(names)} smoke models completed",
            flush=True,
        )
        return

    if args.model == "all":
        parser.error(
            "Full fitting of all models is submitted through submit_fit.sh"
        )

    if args.chains < 1 or args.workers < 1:
        parser.error("--chains and --workers must be positive")
    if args.chain is not None and args.chain < 1:
        parser.error("--chain must be positive")

    chain_ids = (
        [args.chain] if args.chain is not None
        else list(range(1, args.chains + 1))
    )

    run_full(
        study=args.study,
        model_name=args.model,
        chain_ids=chain_ids,
        workers=args.workers,
        samples=samples,
        burn=burn,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
