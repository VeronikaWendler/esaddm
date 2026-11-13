# ESaDDM — Studies 1 and 2

This repository fits hierarchical attentional drift-diffusion models to choice, reaction time, and gaze data from the Eye Tracking Project. It prepares model inputs, fits the models with HDDM, checks convergence and predictive fit, and runs parameter recovery.
Importantly, this is set up to run entirely on a cluster.

## Inputs and setup

Copy the main CSVs from the Eye Tracking Project into the corresponding input folders.

| Study 1 | Study 2 |
|---|---|
| `studies/study1/data/raw/Study1_Behaviour_with_Gaze_AnalysisReady.csv` | `studies/study2/data/raw/Study2_Behaviour_with_Gaze_AnalysisReady.csv` |

For data preparation, use Python 3.9+ and run `python -m pip install -e .` from the repository root. On BlueBEAR, the Slurm scripts use Apptainer and default to `$HOME/containers/hddm_latest.sif`. Export `IMAGE` to use a different HDDM container.

## Run order on BlueBEAR

Run these commands from the repository root. The studies can run in parallel. Wait for each prerequisite job to finish before submitting the next stage. Check `logs/slurm/` for job output.

| Stage | Command | Output |
|---|---|---|
| 1. Prepare or verify inputs | `bash cluster/submit.sh prep both` | Creates `identity_es.csv`, `accuracy_es.csv`, and `accuracy_ee.csv` under each study’s `data/prepared/`, or verifies them if all three exist. |
| 2. Check model inputs | `bash cluster/submit_fit.sh check both all` | Checks the prepared inputs and model designs. For a short test fit, use `bash cluster/submit_fit.sh smoke study1 esaddm_z`. |
| 3. Fit models | `bash cluster/submit_fit.sh run both all` | Submits one job per study and model, with five chains by default. Saves chains and run metadata under `results/fits/<model>/`. |
| 4. Diagnose fits | `bash cluster/submit_diagnose.sh study1 all` and `bash cluster/submit_diagnose.sh study2 all` | Saves convergence statistics, trace plots, posterior summaries, and DIC under `results/diagnostics/<model>/fits/`. |
| 5. Generate posterior predictions | `bash cluster/submit_ppc.sh both all` | Saves posterior predictive simulations under `results/ppc/<model>/`. |
| 6. Plot and compare predictions | `bash cluster/submit_ppc_plot.sh both all` and `bash cluster/submit_ppc_compare.sh both` | Produces per-model tables and plots and a four-model ES comparison under `results/ppc_compare/four_model/`. Both commands use the simulations from step 5 and can run independently. |

For preparation, run `python -m esaddm.cli prepare --study both` on fresh inputs. If the prepared CSVs already exist, use `python -m esaddm.cli verify --study both`. The verifier compares exact bytes, so use Linux or BlueBEAR to verify CSVs saved with Linux line endings.

## Models

The accuracy-coded models, `accuracy_es` and `accuracy_ee`, code responses by whether the higher-value option was chosen and exclude value ties. The four identity-coded ES models share `identity_es.csv` and code responses by whether the symbolic or experiential option was chosen. `addm` and `addm_z` use the shared gaze-weighted value terms. `esaddm` and `esaddm_z` add contrasts for symbolic versus experiential values. The `_z` variants estimate the starting point. See [model specifications](docs/model_specifications.md) for coding rules and sample sizes.

## Follow-up analyses

- **Parameter recovery.** After both full `esaddm_z` fits finish, run `REC_REPS=20 bash cluster/submit_recovery.sh both all` for 20 replicates per study. After those jobs finish, run `python -m esaddm.plot_recovery --study both --expected-reps 20` in a Python environment with Matplotlib. This writes tables and figures to each study’s `results/recovery/esaddm_z/summary/`.
