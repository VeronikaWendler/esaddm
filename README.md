# ESaDDM — clean, reproducible research repository

**Milestone 1: data preparation and verification only.** This repository deliberately
contains **no executable MCMC fitting workflow yet**. The model manifest records the
intended six model specifications, but fitting, diagnostics, PPC and cross-validation
will be added **after** the preparation is validated on BlueBEAR. Nothing in this
repository edits the original `Study1_aDDM` / `Study2_aDDM` directories.

## Layout

```text
esaddm/
├── config/                 study1.json, study2.json, models.json
├── src/esaddm/             ONE shared Python preparation implementation + CLI
├── studies/
│   ├── study1/data/{raw,prepared}/
│   └── study2/data/{raw,prepared}/
├── cluster/                one prep submission wrapper + one sbatch file
├── logs/slurm/             all scheduler logs in one ignored directory
├── verification/           independently derived legacy SHA-256 fingerprints
├── tests/                  automated tests / original-data parity checks
└── docs/                   model semantics and migration instructions
```

## Quick start (local or container)

From the project root, with Python + `numpy` + `pandas`:

```bash
python -m pip install -e '.[test]'
python -m esaddm.cli verify --study both
python -m esaddm.cli prepare --study both
python -m pytest -q
```

If your prepared outputs are already present (as in the **complete bootstrap ZIP**),
run `verify` instead of `prepare`; the latter deliberately refuses to overwrite.

When using the smaller **code-only ZIP**, first copy the original files unchanged:

- `Study1_Behaviour_with_Gaze_AnalysisReady.csv` to `studies/study1/data/raw/`
- `Study2_Behaviour_with_Gaze_AnalysisReady.csv` to `studies/study2/data/raw/`

These original filenames are *deliberately* preserved; downstream outputs are lowercase.

### Output inputs

Each study produces exactly three reusable input tables:

- `data/prepared/accuracy_es.csv` — high-value response coordinate, ES, no value ties
- `data/prepared/accuracy_ee.csv` — high-value response coordinate, EE, no value ties
- `data/prepared/identity_es.csv` — S-upper/E-lower coordinate, ES, ties retained

The four main ES models (`addm`, `addm_z`, `esaddm`, `esaddm_z`) use the same
`identity_es.csv`. All output predictors are lowercase and have no drift intercept
in `config/models.json`. The two accuracy-coded models also have no intercept.

## Data integrity and exact legacy parity

**Every one of the six prepared tables was independently generated using the original
study-specific legacy Python scripts, renamed to lowercase *only*, and hashed.**
The new module was run against both original raw CSVs; **all six generated CSVs
matched these reference SHA-256 fingerprints byte for byte** (after renaming headers).

- The verifier checks raw input file SHA-256 before preparing anything.
- It checks participant count, trial count, column values and canonical CSV SHA-256.
- It never overwrites existing files during `verify`.
- Preparation refuses to replace existing files unless explicitly passed `--force`.
- The original data and generated outputs are listed in `.gitignore` so they're not
  inadvertently committed to Git. The complete bootstrap archive *does* contain
  them to facilitate immediate offline reproduction.
- Every generated dataset has an `.audit.json` with exclusions/filters and hashes.

The fingerprints assume the **specific 2026-10-25 source archive**. Replacing the
raw input file with a revised version appropriately causes validation to stop:
update the comparison only as a documented scientific change, never silently.

## BlueBEAR: staging the clean repo

1. Place the **new project** at a separate location, e.g.
   `/rds/projects/z/zhanglp-vwendler-core/esaddm/`
2. Keep both original AnalysisReady datasets under `studies/study1/data/raw/` and
   `studies/study2/data/raw/`, respectively. All paths resolve relative to this
   unified repository root.
3. Confirm the old image exists (`$HOME/containers/hddm_latest.sif`) or set `IMAGE`.
4. Verify on the login node with the existing environment, or submit preparation:

```bash
./cluster/submit.sh prep both
```

The Slurm launcher **automatically runs `verify`** if all three prepared CSVs already
exist for a study; otherwise, it runs `prepare` on a fresh deployment. If only some
prepared files exist, it **stops safely** instead of silently replacing them.
You can also run `python -m esaddm.cli verify --study both` locally.

`cluster/submit.sh` writes both stdout/stderr to a predictable
`logs/slurm/prep_{study}_{jobid}.log` path. It binds only the new repository root
into the existing Apptainer container. It does not launch any expensive fits.

For computer access without Apptainer, use `PYTHONPATH=src python -m esaddm.cli ...`.

## What's intentionally not implemented yet

HDDM fitting, posterior diagnostics, PPC, group comparisons, CV and figures are not
included in this first milestone. The first aim was to **prove that the rewritten
preparation is equivalent**, not to rewrite expensive models and potentially change
scientific semantics at the same time.

See `docs/model_specifications.md` and `docs/migration.md` for exact coding rules,
tests and the next implementation steps.
