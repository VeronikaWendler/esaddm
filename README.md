# ESaDDM — clean, reproducible research repository

**1: data preparation and verification.** 

## Layout

```text
esaddm/
├── config/                 study1.json, study2.json, models.json
├── src/esaddm/             shared Python preparation implementation + CLI
├── studies/
│   ├── study1/data/{raw,prepared}/
│   └── study2/data/{raw,prepared}/
├── cluster/                one prep submission wrapper + one sbatch file
├── logs/slurm/             all scheduler logs in one directory
├── tests/                  automated tests
└── docs/                   models
```

## Quick start (local or container)

From the project root, with Python + `numpy` + `pandas`:

```bash
python -m pip install -e '.[test]'
python -m esaddm.cli verify --study both
python -m esaddm.cli prepare --study both
python -m pytest -q
```

### Output inputs

Each study produces three input tables:

- `data/prepared/accuracy_es.csv` — high-value response coordinate, ES, no value ties
- `data/prepared/accuracy_ee.csv` — high-value response coordinate, EE, no value ties
- `data/prepared/identity_es.csv` — S-upper/E-lower coordinate, ES, ties retained

The four main ES models (`addm`, `addm_z`, `esaddm`, `esaddm_z`) use the same
`identity_es.csv`. All output predictors are lowercase and have no drift intercept
in `config/models.json`. The two accuracy-coded models also have no intercept.


The Slurm launcher **automatically runs `verify`** if all three prepared CSVs already
exist for a study; otherwise, it runs `prepare` on a fresh deployment. If only some
prepared files exist, it **stops safely** instead of silently replacing them.
You can also run `python -m esaddm.cli verify --study both` locally.

`cluster/submit.sh` writes both stdout/stderr to a
`logs/slurm/prep_{study}_{jobid}.log` path. It binds only the new repository root
into the existing Apptainer container. It does not launch any expensive fits.

For access without Apptainer, use `PYTHONPATH=src python -m esaddm.cli ...`.

## What's intentionally not implemented yet

HDDM fitting, posterior diagnostics, PPC, group comparisons, CV and figures are not
included in this first milestone. The first aim was to **prove that the rewritten
preparation is equivalent**, not to rewrite expensive models and potentially change
scientific semantics at the same time.

See `docs/model_specifications.md` and `docs/migration.md` for exact coding rules,
tests and the next implementation steps.
