"""parameter recovery for esaddm_z from choices and response times.

one replicate is one simulated version of the original ES phase.

steps
-----
1. select generating parameters from a joint empirical posterior draw.
2. calculate each trial's drift using its recorded gaze and value predictors.
3. simulate choices and response times for every trial.
4. refit the same ESaDDM+z model to the simulated data.
5. derive E/S coefficients and attentional ratios from recovered draws.
6. save group and participant recovery estimates and convergence statistics.

ten replicates are run for each study (simila rto Fontanesi et al., 2019).
"""
from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
from scipy.special import expit

from .models import ROOT, build_hddm_model, check_input, load_models

# parameters estimated in the fitted model and recovered here
MODEL = 'esaddm_z'
DIRECT = ('a', 't', 'z', 'b_a', 'b_i', 'delta_a', 'delta_i')

# coefficients and attentional ratios derived from the fitted parameters
DERIVED = ('b_as', 'b_ae', 'b_is', 'b_ie',
           'theta_s', 'theta_e', 'delta_theta_s_minus_e')
RATIOS = ('theta_s', 'theta_e', 'delta_theta_s_minus_e')

# trial-level regressors used in the empirical model
REGRESSORS = ('attention_w', 'inattention_w',
              'attention_contrast', 'inattention_contrast')

# parameter names used by hddm's posterior traces
GROUP_NODES = dict(a='a', t='t', z='z', b_a='v_attention_w',
                   b_i='v_inattention_w', delta_a='v_attention_contrast',
                   delta_i='v_inattention_contrast')

# hddm simulator parameters are ordered as v, a, z, t, sz, sv, st;
# the final three are fixed to zero by the model.
SIM_MODEL = 'full_ddm_hddm_base'

# numerical safeguards and data-format version
NEAR_ZERO_DENOMINATOR = 1e-10
MIN_SOURCE_DRAWS = 50
VERSION = 2

# match the empirical model's mcmc sampling settings
DEFAULT_CHAINS = 5
DEFAULT_SAMPLES = 5000
DEFAULT_BURN = 1000
DEFAULT_PARALLEL_JOBS = 4


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, obj):
    temp = path.with_name(path.name + '.part')
    temp.write_text(json.dumps(obj, indent=2, default=int) + '\n', encoding='utf-8')
    temp.replace(path)


# calculate source-specific parameters
def parameter_dict(base):
    """compute source-specific coefficients and ratios for each posterior draw.

    b_as = b_a + delta_a / 2      b_ae = b_a - delta_a / 2
    b_is = b_i + delta_i / 2      b_ie = b_i - delta_i / 2
    theta_s = b_is / b_as         theta_e = b_ie / b_ae
    """
    a, i, da, di = (np.asarray(base[k], dtype=float)
                    for k in ('b_a', 'b_i', 'delta_a', 'delta_i'))
    b_as, b_ae = a + da / 2, a - da / 2
    b_is, b_ie = i + di / 2, i - di / 2
    with np.errstate(divide='ignore', invalid='ignore'):
        theta_s = np.where(np.abs(b_as) > NEAR_ZERO_DENOMINATOR, b_is / b_as, np.nan)
        theta_e = np.where(np.abs(b_ae) > NEAR_ZERO_DENOMINATOR, b_ie / b_ae, np.nan)
    return dict(b_as=b_as, b_ae=b_ae, b_is=b_is, b_ie=b_ie,
                theta_s=theta_s, theta_e=theta_e,
                delta_theta_s_minus_e=theta_s - theta_e)


# step 1: select generating parameters
def trace_lookup(frame: pd.DataFrame, name: str, subj: int | None):
    """fetch a group or participant posterior trace on its natural scale."""
    candidates = ([name] if subj is None else
                  [f'{name}_subj.{subj}', f'{name}_subj({subj})',
                   f'{name}_subj_{subj}', f'{name}_subj-{subj}'])
    col = next((c for c in candidates if c in frame.columns), None)
    transformed = False
    if col is None and name == 'z':
        candidates = (['z_trans'] if subj is None else
                      [f'z_subj_trans.{subj}', f'z_subj_trans({subj})',
                       f'z_subj_trans_{subj}', f'z_trans_subj.{subj}',
                       f'z_trans_subj({subj})'])
        col = next((c for c in candidates if c in frame.columns), None)
        transformed = col is not None
    if col is None:
        raise KeyError(f'Missing exact group/participant HDDM node {name} for subject {subj}. '
                       f'Candidates: {candidates}. Actual sample: {list(frame.columns)[:35]}')
    out = np.asarray(frame[col], dtype=float)
    if transformed:
        out = expit(out)
    if not len(out) or not np.isfinite(out).all():
        raise ValueError(f'Invalid posterior trace: {col}')
    if name == 'z' and not ((out > 0) & (out < 1)).all():
        raise ValueError('z must be on natural [0,1] scale')
    return out


def get_truth(model, subjects: list[int], *, seed: int):
    """sample one joint posterior iteration for group and participants."""
    frame = model.get_traces()
    rng = np.random.default_rng(seed)
    group = {key: trace_lookup(frame, node, None) for key, node in GROUP_NODES.items()}
    subj = {sid: {key: trace_lookup(frame, node, sid) for key, node in GROUP_NODES.items()}
            for sid in subjects}
    sizes = {len(v) for v in group.values()}
    sizes.update(len(v) for p in subj.values() for v in p.values())
    if len(sizes) != 1 or min(sizes) < MIN_SOURCE_DRAWS:
        raise ValueError(f'Unequal or short source posterior traces: {sizes}')
    pos = int(rng.integers(min(sizes)))
    truth_group = {k: float(v[pos]) for k, v in group.items()}
    truth_subj = {sid: {k: float(v[pos]) for k, v in param.items()}
                  for sid, param in subj.items()}
    for params in [truth_group, *truth_subj.values()]:
        if params['a'] <= 0 or params['t'] < 0 or not 0 < params['z'] < 1:
            raise ValueError('Source posterior has invalid a/t/z parameters')
        params.update({k: float(v) for k, v in parameter_dict(params).items()})
    return truth_group, truth_subj, pos


# steps 2 and 3: simulate data
def make_trial_parameters(design: pd.DataFrame, truth_subj: dict) -> np.ndarray:
    """calculate trial-specific drift and other diffusion parameters.

    v = b_a * attention_w + b_i * inattention_w
        + delta_a * attention_contrast + delta_i * inattention_contrast

    each participant has their own coefficients, a, t, and z.
    """
    keys = list(REGRESSORS)
    betas = np.empty((len(design), 4), dtype=float)
    a, z, t = (np.empty(len(design), dtype=float) for _ in range(3))
    for sid, idx in design.groupby('subj_idx').indices.items():
        p = truth_subj[int(sid)]
        betas[idx, :] = [p['b_a'], p['b_i'], p['delta_a'], p['delta_i']]
        a[idx], z[idx], t[idx] = p['a'], p['z'], p['t']
    v = (design[keys].to_numpy(float) * betas).sum(axis=1)
    from hddm.model_config import model_config
    order = tuple(model_config[SIM_MODEL]['params'])
    expected = ('v', 'a', 'z', 't', 'sz', 'sv', 'st')
    if order != expected:
        raise RuntimeError(f'HDDM simulator parameter order changed: {order}; expected {expected}')
    theta = np.column_stack([v, a, z, t,
                             np.zeros((len(design), 3), dtype=float)])
    if not np.isfinite(theta).all() or (theta[:, 1] <= 0).any() or (theta[:, 3] < 0).any():
        raise ValueError('Invalid trial-level DDM simulation parameters')
    return theta.astype('float32')


def _draw_ddm(theta: np.ndarray):
    """simulate choices and response times using hddm."""
    from hddm.simulators import simulator
    raw = simulator(theta=theta, model=SIM_MODEL, n_samples=1, max_t=20)
    rts = np.asarray(raw[0], dtype=float).reshape(-1)
    responses = np.asarray(raw[1]).reshape(-1)
    if len(rts) != len(theta) or len(responses) != len(theta):
        raise RuntimeError(f'Simulator output length mismatch: {len(rts)}, {len(responses)}, {len(theta)}')
    # response 1 indicates s; response 0 indicates e.
    if not np.isin(responses, (0, 1)).all():
        raise RuntimeError(f'Simulator returned unexpected responses: {np.unique(responses)}')
    if not np.isfinite(rts).all():
        raise RuntimeError('Non-finite simulated RT')
    return np.abs(rts), responses.astype(int)


def generate_synthetic(design: pd.DataFrame, truth_subj: dict, *, seed: int,
                       min_rt: float):
    """simulate a choice and response time for every trial.

    response times below the study's inclusion threshold are redrawn so
    each simulated experiment has the same trial count as the original.
    """
    theta = make_trial_parameters(design, truth_subj)
    np.random.seed(seed)  # HDDM's simulator consumes numpy's RNG
    rt, response = _draw_ddm(theta)
    mask = rt < min_rt
    rounds = 0
    while np.any(mask):
        rounds += 1
        if rounds > 25:
            raise RuntimeError('RT-cutoff redraw did not converge after 25 rounds')
        ix = np.flatnonzero(mask)
        rt[ix], response[ix] = _draw_ddm(theta[ix])
        mask = rt < min_rt
    result = design[['subj_idx', *REGRESSORS]].copy().reset_index(drop=True)
    result['rt'] = rt
    result['response'] = response
    if not ((result.rt >= min_rt).all() and result.response.isin((0, 1)).all()):
        raise RuntimeError('Invalid synthetic DDM dataset')
    return result[['subj_idx', 'rt', 'response', *REGRESSORS]], rounds


# step 4: fit the model
def extract_trace_matrix(model, subjects: list[int]):
    traces = model.get_traces()
    data = {}
    for key, node in GROUP_NODES.items():
        data['group__' + key] = trace_lookup(traces, node, None)
        for sid in subjects:
            data[f'subj{sid}__{key}'] = trace_lookup(traces, node, sid)
    if len({len(v) for v in data.values()}) != 1:
        raise RuntimeError('Inconsistent MCMC posterior array lengths')
    return data


def fit_chain(study: str, rep: int, chain: int, synthetic_csv: str,
              out: str, subjects: list[int], samples: int, burn: int,
              seed: int, spec):
    """fit one chain with the empirical model specification.

    save parameter traces so completed chains can be reused if a job fails.
    """
    import hddm
    out_path = Path(out)
    if out_path.exists():
        raise FileExistsError(f'Recovery chain trace already exists: {out_path}')
    np.random.seed(seed)
    data = pd.read_csv(synthetic_csv)
    if data.subj_idx.nunique() != len(subjects):
        raise RuntimeError('Synthetic participant count mismatch')
    model = build_hddm_model(data, spec, hddm_module=hddm)
    model.find_starting_values()
    print(f'fit {study}/rep{rep:02d}/chain{chain}: samples={samples} burn={burn}', flush=True)
    model.sample(samples, burn=burn)  # sample into memory
    arrays = extract_trace_matrix(model, subjects)
    retained = {len(v) for v in arrays.values()}
    if retained != {samples - burn}:
        raise RuntimeError(f'Unexpected MCMC retained sample count {retained}; expected {samples - burn}')
    temp = out_path.with_name(out_path.name + '.part')
    with temp.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
    temp.replace(out_path)
    print(f'saved recovery chain {study}/rep{rep:02d}/{chain}: {out_path.name}', flush=True)
    del model
    gc.collect()
    return str(out_path)


# steps 5 and 6: derive parameters and summarise recovery
def short_hdi(x: np.ndarray, mass: float = .95):
    vals = np.sort(np.asarray(x, float)[np.isfinite(x)])
    if len(vals) < 20:
        raise ValueError('Too few valid draws for recovery interval')
    width = int(np.floor(mass * len(vals)))
    span = vals[width:] - vals[:-width]
    lo = int(np.argmin(span))
    return float(vals[lo]), float(vals[lo + width])


def summarise_chains(traces: list[dict], group_true: dict,
                     subj_true: dict, subjects: list[int],
                     study: str, rep: int):
    """summarise recovery for each group and participant parameter.

    save posterior means, medians, 95% hdis, r-hat, and effective sample sizes.
    theta values are computed draw by draw; denominator checks flag unstable
    ratios without interrupting the recovery of directly fitted parameters.
    """
    rows = []
    levels = [('group', None, group_true, 'group__')]
    levels += [('individual', sid, subj_true[sid], f'subj{sid}__')
               for sid in subjects]

    for scope, sid, p_true, prefix in levels:
        # keep the same posterior index across coefficients
        direct_chains = [{k: np.asarray(chain[prefix + k], float)
                          for k in DIRECT} for chain in traces]
        derived_chains = [parameter_dict(chain) for chain in direct_chains]

        for key in (*DIRECT, *DERIVED):
            is_ratio = key in RATIOS
            arrays = [np.asarray((d if key in DIRECT else r)[key], float)
                      for d, r in zip(direct_chains, derived_chains)]
            if len({len(x) for x in arrays}) != 1:
                raise RuntimeError(f'Unequal traces for {scope}/{sid}/{key}')

            matrix = np.stack(arrays)
            finite_mask = np.isfinite(matrix)
            fraction_finite = float(finite_mask.mean())
            if not is_ratio and not finite_mask.all():
                raise ValueError(f'Nonfinite primary parameter: {scope}/{sid}/{key}')
            flat = matrix[finite_mask]

            # check whether theta denominators are close to zero
            denominator_ci95_contains_zero = False
            if key in RATIOS:
                denominator_keys = (('b_as',) if key == 'theta_s' else
                                    ('b_ae',) if key == 'theta_e' else
                                    ('b_as', 'b_ae'))
                for dkey in denominator_keys:
                    denominator_draws = np.concatenate(
                        [np.asarray(r[dkey], float) for r in derived_chains])
                    q025, q975 = np.quantile(denominator_draws, [.025, .975])
                    if q025 <= 0 <= q975:
                        denominator_ci95_contains_zero = True

            warnings = []
            if fraction_finite < 1.0:
                warnings.append('some_ratios_undefined')
            if denominator_ci95_contains_zero:
                warnings.append('attended_denominator_ci95_contains_zero')
            truth = float(p_true[key])
            if not np.isfinite(truth):
                warnings.append('true_ratio_undefined')

            if len(flat) >= 20:
                lo, hi = short_hdi(flat)
                recovered_mean = float(flat.mean())
                recovered_median = float(np.median(flat))
                coverage = (bool(lo <= truth <= hi) if np.isfinite(truth)
                            else np.nan)
            elif is_ratio:
                # save missing values when the ratio is undefined
                lo = hi = recovered_mean = recovered_median = np.nan
                coverage = np.nan
                warnings.append('insufficient_defined_ratio_draws')
            else:
                raise ValueError(f'Too few posterior draws for {scope}/{sid}/{key}')

            # convergence statistics require finite draws across all chains
            if finite_mask.all():
                try:
                    rhat = float(np.asarray(az.rhat(matrix)))
                    bulk = float(np.asarray(az.ess(matrix, method='bulk')))
                    tail = float(np.asarray(az.ess(matrix, method='tail', prob=(.05, .95))))
                except Exception as exc:
                    if not is_ratio:
                        raise RuntimeError(f'ArviZ failed for {scope}/{sid}/{key}') from exc
                    rhat = bulk = tail = np.nan
                    warnings.append('ratio_convergence_not_computable')
            else:
                rhat = bulk = tail = np.nan

            rows.append(dict(
                study=study, rep=rep, scope=scope,
                subj_idx=sid if sid is not None else np.nan,
                parameter=key, kind=('direct' if key in DIRECT else
                                     'derived_ratio' if is_ratio else 'derived_coefficient'),
                true=truth, recovered_mean=recovered_mean,
                recovered_median=recovered_median,
                hdi_lower=lo, hdi_upper=hi, covered_95=coverage,
                r_hat=rhat, ess_bulk=bulk, ess_tail=tail,
                n_posterior_draws=int(matrix.size),
                fraction_finite_posterior_draws=fraction_finite,
                denominator_ci95_contains_zero=(denominator_ci95_contains_zero
                                                 if is_ratio else np.nan),
                ratio_warning=';'.join(warnings),
            ))

    result = pd.DataFrame(rows)
    # assess convergence for fitted parameters at group and participant levels
    direct = result[result.kind.eq('direct')]
    converged = (direct.r_hat.le(1.01) & direct.ess_bulk.ge(400)
                 & direct.ess_tail.ge(400)).all()
    return result, bool(converged)


def launch_independent_chain_processes(study: str, rep: int, remaining: list,
                                       *, root: Path, chains: int, samples: int,
                                       burn: int, seed: int, jobs: int):
    """run up to `jobs` recovery chains in separate python processes.

    saved traces allow a failed job to resume without refitting completed chains.
    """
    if not remaining:
        return
    if jobs < 2:
        raise ValueError('Use direct fit_chain for single-process recovery')

    queue = list(remaining)
    launched = []
    active = []
    try:
        while queue or active:
            while queue and len(active) < jobs:
                chain, path = queue.pop(0)
                command = [sys.executable, '-m', 'esaddm.recovery',
                           '--study', study, '--rep', str(rep),
                           '--chains', str(chains), '--samples', str(samples),
                           '--burn', str(burn), '--seed', str(seed),
                           '--root', str(root), '--worker-chain', str(chain)]
                print(f'start chain '
                      f'{study}/rep{rep:02d}/chain{chain}', flush=True)
                process = subprocess.Popen(command, cwd=str(root))
                item = (chain, Path(path), process)
                active.append(item)
                launched.append(item)

            for item in list(active):
                chain, path, process = item
                returncode = process.poll()
                if returncode is None:
                    continue
                if returncode != 0:
                    raise RuntimeError(
                        f'HDDM chain {study}/rep{rep:02d}/{chain} failed '
                        f'(exit status {returncode}); see the Slurm stderr. '
                        'Completed chain traces remain available for resumption.')
                if not path.is_file():
                    raise RuntimeError(f'HDDM worker exited without saved trace: {path}')
                print(f'finished chain '
                      f'{study}/rep{rep:02d}/chain{chain}', flush=True)
                active.remove(item)

            if active:
                time.sleep(2)
    except BaseException:
        for _, _, process in launched:
            if process.poll() is None:
                process.terminate()
        for _, _, process in launched:
            if process.poll() is None:
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
            process.wait()
        raise


def run_single_chain_worker(study: str, rep: int, chain: int, *,
                            root: Path, chains: int, samples: int, burn: int,
                            seed: int):
    """fit a single chain using its saved synthetic dataset."""
    if study not in ('study1', 'study2') or not 1 <= chain <= chains:
        raise ValueError('Invalid chain or study for worker')
    rep_dir = root / 'studies' / study / 'results' / 'recovery' / MODEL / f'rep_{rep:02d}'
    synthetic_path = rep_dir / 'synthetic.csv.gz'
    truth_path = rep_dir / 'truth.json'
    trace_path = rep_dir / 'temporary_chains' / f'chain{chain}.npz'
    if not synthetic_path.is_file() or not truth_path.is_file():
        raise FileNotFoundError('Missing simulation inputs for recovery chain worker')
    truth = json.loads(truth_path.read_text(encoding='utf-8'))
    expected = dict(chains=chains, samples=samples, burn=burn, seed=seed)
    if (truth.get('study') != study or truth.get('rep') != rep
            or truth.get('version') != VERSION
            or truth.get('recovery_config') != expected):
        raise ValueError('Worker arguments do not match saved generating experiment')
    subjects = sorted(int(k) for k in truth['generating_individual'])
    spec = load_models(root)[MODEL]
    if (spec.coordinate != 's_upper_e_lower' or not spec.estimate_z
            or tuple(spec.predictors) != REGRESSORS):
        raise ValueError('Not the registered ESaDDM+z model')
    fit_chain(study, rep, chain, str(synthetic_path), str(trace_path), subjects,
              samples, burn, seed + rep * 100000 + chain, spec)



# run a complete recovery replicate
def run_rep(study: str, rep: int, *, reps: int = 10, chains: int = DEFAULT_CHAINS,
            samples: int = DEFAULT_SAMPLES, burn: int = DEFAULT_BURN,
            jobs: int = DEFAULT_PARALLEL_JOBS, seed: int = 20261026,
            root: Path = ROOT, keep_traces: bool = False):
    if study not in ('study1', 'study2') or rep < 0 or rep >= reps:
        raise ValueError('Invalid study/rep ID')
    if not 2 <= chains <= 8 or not (200 <= samples and 0 <= burn < samples):
        raise ValueError('Invalid chain or sampling configuration')
    if jobs < 1 or jobs > chains:
        raise ValueError('jobs must be from 1 to number of chains')
    print(f'step 0: {study} synthetic experiment {rep + 1}/{reps}: '
          f'ESaDDM+z; {chains} chains x {samples} iterations '
          f'({burn} burn-in); parallel workers={jobs}', flush=True)
    import hddm
    spec = load_models(root)[MODEL]
    assert spec.coordinate == 's_upper_e_lower' and spec.estimate_z
    assert tuple(spec.predictors) == REGRESSORS
    empirical, audit = check_input(study, spec, root)
    subjects = sorted(int(s) for s in empirical.subj_idx.unique())
    rep_dir = root / 'studies' / study / 'results' / 'recovery' / MODEL / f'rep_{rep:02d}'
    finished = rep_dir / 'complete.json'
    if finished.exists():
        previous = json.loads(finished.read_text(encoding='utf-8'))
        if not (rep_dir / 'truth.json').is_file():
            raise FileNotFoundError('Missing truth.json for completed recovery')
        saved_truth = json.loads((rep_dir / 'truth.json').read_text(encoding='utf-8'))
        if saved_truth.get('version') != VERSION:
            raise ValueError('Saved recovery data have an incompatible format version')
        same = (previous.get('status') == 'completed'
                and previous.get('study') == study
                and previous.get('model') == MODEL
                and previous.get('rep') == rep
                and previous.get('n_recovery_chains') == chains
                and previous.get('samples_per_chain') == samples
                and previous.get('burn_per_chain') == burn
                and previous.get('seed') == seed)
        if not same:
            raise FileExistsError(f'Existing completed recovery has different settings: {finished}')
        if not (rep_dir / 'recovery_parameters.csv').is_file():
            raise FileNotFoundError('Completed recovery manifest without results table')
        print(f'skip completed recovery {study}/rep{rep:02d}; saved data retained', flush=True)
        return rep_dir
    rep_dir.mkdir(parents=True, exist_ok=True)
    synth_path = rep_dir / 'synthetic.csv.gz'
    truth_path = rep_dir / 'truth.json'
    traces_dir = rep_dir / 'temporary_chains'
    traces_dir.mkdir(exist_ok=True)
    trace_paths = [traces_dir / f'chain{i}.npz' for i in range(1, chains + 1)]
    if synth_path.exists() != truth_path.exists():
        raise ValueError('Partial truth/synthetic inputs: archive replicate before retrying')
    if not synth_path.exists():
        full_fit_manifest = (root / 'studies' / study / 'results' / 'fits' / MODEL
                             / f'{MODEL}_{study}_run.json')
        if not full_fit_manifest.is_file():
            raise FileNotFoundError(f'Missing full-fit completion manifest: {full_fit_manifest}')
        fm = json.loads(full_fit_manifest.read_text(encoding='utf-8'))
        if (fm.get('status') != 'completed' or fm.get('study') != study
                or fm.get('model') != MODEL or len(fm.get('chains', [])) != 5):
            raise ValueError(f'Full empirical five-chain fit not complete: {full_fit_manifest}')
        source_chain = (rep % 5) + 1
        source = root / 'studies' / study / 'results' / 'fits' / MODEL / f'{MODEL}_{study}_chain{source_chain}.hddm'
        if not source.is_file():
            raise FileNotFoundError(f'Missing empirical posterior chain: {source}')
        print(f'step 1: select generating parameters from {source}', flush=True)
        fitted = hddm.load(str(source))
        group, individual, pos = get_truth(fitted, subjects, seed=seed + rep * 10007)
        del fitted
        gc.collect()
        print('steps 2-3: calculate trial drift and simulate choices and response times', flush=True)
        synthetic, regen_rounds = generate_synthetic(
            empirical, individual, seed=seed + rep * 10007 + 347,
            min_rt=float(json.loads((root / 'config' / f'{study}.json').read_text())['min_rt_seconds']))
        synthetic.to_csv(synth_path, index=False, compression='gzip', float_format='%.10g')
        meta = dict(study=study, model=MODEL, rep=rep, version=VERSION,
                    seed=seed,
                    source_chain=source_chain, source_posterior_index=pos,
                    empirical_trials=len(empirical), participants=len(subjects),
                    source_audit=audit, generating_group=group,
                    generating_individual={str(s): individual[s] for s in subjects},
                    cutoff_resimulation_rounds=regen_rounds,
                    recovery_model_formula=spec.drift_formula,
                    recovery_config=dict(chains=chains, samples=samples, burn=burn, seed=seed))
        write_json(truth_path, meta)
        print(f'simulated {study}/rep{rep:02d}: {len(synthetic)} trials', flush=True)
    truth = json.loads(truth_path.read_text())
    if (truth.get('study') != study or truth.get('rep') != rep or
            truth.get('version') != VERSION):
        raise ValueError('Saved recovery dataset does not match the requested study, replicate, or format version')
    expected_config = dict(chains=chains, samples=samples, burn=burn, seed=seed)
    if truth.get('recovery_config') != expected_config:
        raise ValueError('Saved recovery dataset uses different mcmc settings; use the original settings or a new replicate')
    group = truth['generating_group']
    individual = {int(k): v for k, v in truth['generating_individual'].items()}
    if set(individual) != set(subjects):
        raise ValueError('Synthetic source participant IDs mismatch')
    chk = pd.read_csv(synth_path, usecols=['subj_idx', 'rt', 'response'])
    min_rt = float(json.loads((root / 'config' / f'{study}.json').read_text())['min_rt_seconds'])
    if (len(chk) != len(empirical) or set(chk.subj_idx.astype(int)) != set(subjects)
            or (chk.rt < min_rt - 1e-8).any() or not chk.response.isin((0, 1)).all()):
        raise ValueError('Synthetic trial file failed recovery/resumption checks')
    print('step 4: fit ESaDDM+z to simulated data', flush=True)
    if not all(p.is_file() for p in trace_paths):
        remaining = [(chain, str(path)) for chain, path in enumerate(trace_paths, 1)
                     if not path.is_file()]
        if jobs == 1 or len(remaining) == 1:
            for chain, path in remaining:
                fit_chain(study, rep, chain, str(synth_path), path,
                          subjects, samples, burn, seed + rep * 100000 + chain,
                          spec)
        else:
            launch_independent_chain_processes(
                study, rep, remaining, root=root, chains=chains,
                samples=samples, burn=burn, seed=seed, jobs=jobs)
    loaded = []
    for path in trace_paths:
        with np.load(path, allow_pickle=False) as f:
            loaded.append({key: f[key] for key in f.files})
    print('steps 5-6: calculate derived parameters and recovery summaries', flush=True)
    recovered, converged = summarise_chains(loaded, group, individual, subjects, study, rep)
    output = rep_dir / 'recovery_parameters.csv'
    temp = output.with_name(output.name + '.part')
    recovered.to_csv(temp, index=False, float_format='%.10g')
    temp.replace(output)
    record = dict(study=study, model=MODEL, rep=rep, status='completed',
                  n_synthetic_trials=len(empirical), subjects=len(subjects),
                  n_recovery_chains=chains, samples_per_chain=samples, burn_per_chain=burn,
                  total_retained_draws=chains * (samples - burn), seed=seed,
                  n_summary_rows=len(recovered), direct_convergence_ok=converged,
                  max_direct_rhat=float(recovered.loc[recovered.kind.eq('direct'), 'r_hat'].max()),
                  min_direct_ess_bulk=float(recovered.loc[recovered.kind.eq('direct'), 'ess_bulk'].min()),
                  min_direct_ess_tail=float(recovered.loc[recovered.kind.eq('direct'), 'ess_tail'].min()),
                  finished_utc=utc_now())
    write_json(finished, record)
    print('step 7: save recovery results', flush=True)
    # retain chain traces when convergence diagnostics need review
    if not keep_traces and converged:
        for path in trace_paths:
            path.unlink()
        traces_dir.rmdir()
    print(f'finished recovery {study}/rep{rep:02d}: convergence={converged}; '
          f'{len(recovered)} parameter rows; no plots', flush=True)
    if not converged:
        print('convergence check not met for every parameter; traces retained', flush=True)
    return rep_dir


def main(argv=None):
    p = argparse.ArgumentParser(description='Simulate and fit ESaDDM+z parameter recovery data')
    p.add_argument('--study', choices=['study1','study2'], required=True)
    p.add_argument('--rep', type=int, required=True, help='Zero-indexed replicate, 0..9')
    p.add_argument('--reps', type=int, default=10)
    p.add_argument('--chains', type=int, default=DEFAULT_CHAINS,
                   help='MCMC chains; empirical default is 5')
    p.add_argument('--samples', type=int, default=DEFAULT_SAMPLES,
                   help='Iterations PER chain including burn-in (empirical: 5000)')
    p.add_argument('--burn', type=int, default=DEFAULT_BURN,
                   help='First 1000 MCMC iterations of each chain discarded')
    p.add_argument('--jobs', type=int, default=DEFAULT_PARALLEL_JOBS,
                   help='Max independent chains running simultaneously')
    p.add_argument('--seed', type=int, default=20261026)
    p.add_argument('--keep-traces', action='store_true')
    p.add_argument('--root', type=Path, default=ROOT, help=argparse.SUPPRESS)
    p.add_argument('--worker-chain', type=int, default=None, help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    if a.worker_chain is not None:
        run_single_chain_worker(
            a.study, a.rep, a.worker_chain, root=a.root,
            chains=a.chains, samples=a.samples, burn=a.burn, seed=a.seed)
    else:
        run_rep(a.study, a.rep, reps=a.reps, chains=a.chains,
                samples=a.samples, burn=a.burn, jobs=a.jobs,
                seed=a.seed, root=a.root, keep_traces=a.keep_traces)


if __name__ == '__main__':
    main()
