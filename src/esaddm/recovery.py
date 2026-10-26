"""Posterior-informed ESaDDM+z parameter-recovery generation and fitting.

Independent per-study/per-rep jobs. This module does NOT create figures.
For each replicate it draws coherent group + participant parameters from one
saved empirical joint posterior iteration, simulates the prepared ES design,
refits the EXACT registered model, and saves true-vs-recovered summaries.

HDDM MCMC lives in memory; only compact chain traces are retained temporarily
so a failed job can be resumed without refitting completed recovery chains.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
from scipy.special import expit

from .models import ROOT, build_hddm_model, check_input, load_models

MODEL = 'esaddm_z'
DIRECT = ('a', 't', 'z', 'b_a', 'b_i', 'delta_a', 'delta_i')
DERIVED = ('b_as', 'b_ae', 'b_is', 'b_ie')  # ratios are exploratory; not primary recovery targets
REGRESSORS = ('attention_w', 'inattention_w', 'attention_contrast', 'inattention_contrast')
GROUP_NODES = dict(a='a', t='t', z='z', b_a='v_attention_w',
                   b_i='v_inattention_w', delta_a='v_attention_contrast',
                   delta_i='v_inattention_contrast')
SIM_MODEL = 'full_ddm_hddm_base'
CONTAMINATION = .05
OUTLIER_SIGNED_RANGE = (-5., 5.)  # signed uniform density = 0.1 per second
VERSION = 1


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, obj):
    temp = path.with_name(path.name + '.part')
    temp.write_text(json.dumps(obj, indent=2, default=int) + '\n', encoding='utf-8')
    temp.replace(path)


def parameter_dict(base):
    """Compute source-specific parameters from aligned ESaDDM coefficients."""
    a, i, da, di = (np.asarray(base[k], dtype=float)
                    for k in ('b_a', 'b_i', 'delta_a', 'delta_i'))
    b_as, b_ae = a + da / 2, a - da / 2
    b_is, b_ie = i + di / 2, i - di / 2
    with np.errstate(divide='ignore', invalid='ignore'):
        theta_s = np.where(np.abs(b_as) > 1e-10, b_is / b_as, np.nan)
        theta_e = np.where(np.abs(b_ae) > 1e-10, b_ie / b_ae, np.nan)
    return dict(b_as=b_as, b_ae=b_ae, b_is=b_is, b_ie=b_ie,
                theta_s=theta_s, theta_e=theta_e,
                delta_theta_s_minus_e=theta_s - theta_e)


def trace_lookup(frame: pd.DataFrame, name: str, subj: int | None):
    """Fetch group or participant posterior trace on natural scale."""
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
    frame = model.get_traces()
    rng = np.random.default_rng(seed)
    group = {key: trace_lookup(frame, node, None) for key, node in GROUP_NODES.items()}
    subj = {sid: {key: trace_lookup(frame, node, sid) for key, node in GROUP_NODES.items()}
            for sid in subjects}
    sizes = {len(v) for v in group.values()}
    sizes.update(len(v) for p in subj.values() for v in p.values())
    if len(sizes) != 1 or min(sizes) < 50:
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


def make_trial_parameters(design: pd.DataFrame, truth_subj: dict) -> np.ndarray:
    """Vectorised trial-dependent DDM parameters in HDDM simulator order."""
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
    """Same SSMS/HDDM regression simulator as HDDM's wfpt_reg.random."""
    from hddm.simulators import simulator
    raw = simulator(theta=theta, model=SIM_MODEL, n_samples=1, max_t=20)
    rts = np.asarray(raw[0], dtype=float).reshape(-1)
    responses = np.asarray(raw[1]).reshape(-1)
    if len(rts) != len(theta) or len(responses) != len(theta):
        raise RuntimeError(f'Simulator output length mismatch: {len(rts)}, {len(responses)}, {len(theta)}')
    # full_ddm_hddm_base defines choices [0,1], not -1/+1.
    if not np.isin(responses, (0, 1)).all():
        raise RuntimeError(f'Simulator returned unexpected responses: {np.unique(responses)}')
    if not np.isfinite(rts).all():
        raise RuntimeError('Non-finite simulated RT')
    return np.abs(rts), responses.astype(int)


def generate_synthetic(design: pd.DataFrame, truth_subj: dict, *, seed: int,
                       min_rt: float, p_outlier: float = CONTAMINATION):
    """Simulate conditional choice/RT with fixed empirical gaze/value predictors.

    Draw 5% signed-RT uniform contamination, matching HDDM's w_outlier=.1
    convention; re-draw too-fast simulated trials to match the empirical RT
    selection rule. This conditioning changes the retained mixture rate slightly.
    """
    theta = make_trial_parameters(design, truth_subj)
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    n = len(theta)
    outlier = rng.random(n) < p_outlier
    rt, response = _draw_ddm(theta)
    signed = rng.uniform(*OUTLIER_SIGNED_RANGE, size=n)
    rt[outlier] = np.abs(signed[outlier])
    response[outlier] = (signed[outlier] > 0).astype(int)
    mask = rt < min_rt
    rounds = 0
    while np.any(mask):
        rounds += 1
        if rounds > 25:
            raise RuntimeError('Too many RT-cutoff regeneration rounds')
        ix = np.flatnonzero(mask)
        new_rt, new_response = _draw_ddm(theta[ix])
        contam = rng.random(len(ix)) < p_outlier
        signed = rng.uniform(*OUTLIER_SIGNED_RANGE, size=len(ix))
        new_rt[contam] = np.abs(signed[contam])
        new_response[contam] = (signed[contam] > 0).astype(int)
        rt[ix], response[ix], outlier[ix] = new_rt, new_response, contam
        mask = rt < min_rt
    result = design[['subj_idx', *REGRESSORS]].copy().reset_index(drop=True)
    result['rt'] = rt
    result['response'] = response
    if not ((result.rt >= min_rt).all() and result.response.isin((0, 1)).all()):
        raise RuntimeError('Invalid synthetic dataset')
    return result[['subj_idx','rt','response', *REGRESSORS]], int(outlier.sum()), rounds


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
              seed: int, spec, root_str: str):
    """One independent HDDM chain. Saves only compact trace arrays."""
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
    print(f'FIT {study}/rep{rep:02d}/chain{chain}: samples={samples} burn={burn}', flush=True)
    model.sample(samples, burn=burn)  # RAM database, no redundant HDDM/PyMC files
    arrays = extract_trace_matrix(model, subjects)
    retained = {len(v) for v in arrays.values()}
    if retained != {samples - burn}:
        raise RuntimeError(f'Unexpected MCMC retained sample count {retained}; expected {samples - burn}')
    temp = out_path.with_name(out_path.name + '.part')
    with temp.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
    temp.replace(out_path)
    print(f'PASS recovery chain {study}/rep{rep:02d}/{chain}: {out_path.name}', flush=True)
    del model
    gc.collect()
    return str(out_path)


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
    """One row per parameter / person, using joint-index derived posterior draws."""
    rows = []
    for scope, sid, p_true, prefix in [
        ('group', None, group_true, 'group__'),
        *[('individual', sid, subj_true[sid], f'subj{sid}__') for sid in subjects]
    ]:
        direct_chains = [{k: np.asarray(chain[prefix + k], float) for k in DIRECT}
                         for chain in traces]
        derived_chains = [parameter_dict(chain) for chain in direct_chains]
        for key in (*DIRECT, *DERIVED):
            arrays = [np.asarray((d if key in DIRECT else r)[key], float)
                      for d, r in zip(direct_chains, derived_chains)]
            if len({len(x) for x in arrays}) != 1:
                raise RuntimeError(f'Unequal trace lengths for {scope} {sid} {key}')
            if any(not np.isfinite(a).all() for a in arrays):
                raise ValueError(f'Nonfinite derived trace {scope} {sid} {key}; check ratio denominators')
            matrix = np.stack(arrays)
            flat = matrix.reshape(-1)
            lo, hi = short_hdi(flat)
            try:
                rhat = float(np.asarray(az.rhat(matrix)))
                bulk = float(np.asarray(az.ess(matrix, method='bulk')))
                tail = float(np.asarray(az.ess(matrix, method='tail', prob=(.05, .95))))
            except Exception as exc:
                raise RuntimeError(f'ArviZ failed for {scope}/{sid}/{key}') from exc
            truth = float(p_true[key])
            rows.append(dict(study=study, rep=rep, scope=scope,
                             subj_idx=sid if sid is not None else np.nan,
                             parameter=key, kind='direct' if key in DIRECT else 'derived',
                             true=truth, recovered_mean=float(flat.mean()),
                             recovered_median=float(np.median(flat)),
                             hdi_lower=lo, hdi_upper=hi,
                             covered_95=bool(lo <= truth <= hi),
                             r_hat=rhat, ess_bulk=bulk, ess_tail=tail,
                             n_posterior_draws=len(flat)))
    result = pd.DataFrame(rows)
    direct = result[result.kind.eq('direct')]
    converged = (direct.r_hat.le(1.01) & direct.ess_bulk.ge(400)
                 & direct.ess_tail.ge(400)).all()
    return result, bool(converged)


def run_rep(study: str, rep: int, *, reps: int = 10, chains: int = 4,
            samples: int = 4000, burn: int = 1000, jobs: int = 4,
            seed: int = 20261026, root: Path = ROOT, keep_traces: bool = False):
    if study not in ('study1', 'study2') or rep < 0 or rep >= reps:
        raise ValueError('Invalid study/rep ID')
    if not 2 <= chains <= 8 or not (200 <= samples and 0 <= burn < samples):
        raise ValueError('Invalid chain or sampling configuration')
    if jobs < 1 or jobs > chains:
        raise ValueError('jobs must be from 1 to number of chains')
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
        print(f'SKIP completed recovery {study}/rep{rep:02d}; saved data retained', flush=True)
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
        print(f'LOAD EMPIRICAL {study}: {source}', flush=True)
        fitted = hddm.load(str(source))
        group, individual, pos = get_truth(fitted, subjects, seed=seed + rep * 10007)
        del fitted
        gc.collect()
        synthetic, n_outlier, regen_rounds = generate_synthetic(
            empirical, individual, seed=seed + rep * 10007 + 347,
            min_rt=float(json.loads((root / 'config' / f'{study}.json').read_text())['min_rt_seconds']))
        synthetic.to_csv(synth_path, index=False, compression='gzip', float_format='%.10g')
        meta = dict(study=study, model=MODEL, rep=rep, version=VERSION, seed=seed,
                    source_chain=source_chain, source_posterior_index=pos,
                    empirical_trials=len(empirical), participants=len(subjects),
                    source_audit=audit, generating_group=group,
                    generating_individual={str(s): individual[s] for s in subjects},
                    injected_outliers_retained=n_outlier, cutoff_resimulation_rounds=regen_rounds,
                    nominal_outlier_probability=CONTAMINATION,
                    signed_outlier_range=list(OUTLIER_SIGNED_RANGE),
                    recovery_model_formula=spec.drift_formula,
                    recovery_config=dict(chains=chains, samples=samples, burn=burn, seed=seed))
        write_json(truth_path, meta)
        print(f'PASS SIMULATION {study}/rep{rep:02d}: {len(synthetic)} trials', flush=True)
    truth = json.loads(truth_path.read_text())
    if truth['study'] != study or truth['rep'] != rep or truth['version'] != VERSION:
        raise ValueError('Mismatched synthetic input manifest')
    expected_config = dict(chains=chains, samples=samples, burn=burn, seed=seed)
    if truth.get('recovery_config') != expected_config:
        raise ValueError('Recovery settings changed from saved synthetic dataset. '
                         'Restore the original settings or archive the replicate before rerunning.')
    group = truth['generating_group']
    individual = {int(k): v for k, v in truth['generating_individual'].items()}
    if set(individual) != set(subjects):
        raise ValueError('Synthetic source participant IDs mismatch')
    chk = pd.read_csv(synth_path, usecols=['subj_idx', 'rt', 'response'])
    min_rt = float(json.loads((root / 'config' / f'{study}.json').read_text())['min_rt_seconds'])
    if (len(chk) != len(empirical) or set(chk.subj_idx.astype(int)) != set(subjects)
            or (chk.rt < min_rt - 1e-8).any() or not chk.response.isin((0, 1)).all()):
        raise ValueError('Synthetic trial file failed recovery/resumption checks')
    if not all(p.is_file() for p in trace_paths):
        remaining = [(chain, str(path)) for chain, path in enumerate(trace_paths, 1)
                     if not path.is_file()]
        if jobs == 1 or len(remaining) == 1:
            for chain, path in remaining:
                fit_chain(study, rep, chain, str(synth_path), path,
                          subjects, samples, burn, seed + rep * 100000 + chain,
                          spec, str(root))
        else:
            from joblib import Parallel, delayed
            Parallel(n_jobs=min(jobs, len(remaining)), backend='loky')(
                delayed(fit_chain)(study, rep, chain, str(synth_path), path,
                                   subjects, samples, burn, seed + rep * 100000 + chain,
                                   spec, str(root)) for chain, path in remaining)
    loaded = []
    for path in trace_paths:
        with np.load(path, allow_pickle=False) as f:
            loaded.append({key: f[key] for key in f.files})
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
    if not keep_traces:
        for path in trace_paths:
            path.unlink()
        traces_dir.rmdir()
    print(f'PASS RECOVERY {study}/rep{rep:02d}: convergence={converged}; '
          f'{len(recovered)} parameter rows; no plots', flush=True)
    if not converged:
        print('CAUTION: Some parameters did not meet Rhat<=1.01 / ESS>=400; '
              'inspect recovery_parameters.csv, do not hide those replicates.', flush=True)
    return rep_dir


def main(argv=None):
    p = argparse.ArgumentParser(description='Generate/refit ESaDDM+z recovery datasets; NO PLOTS')
    p.add_argument('--study', choices=['study1','study2'], required=True)
    p.add_argument('--rep', type=int, required=True, help='Zero-indexed replicate, 0..9')
    p.add_argument('--reps', type=int, default=10)
    p.add_argument('--chains', type=int, default=4)
    p.add_argument('--samples', type=int, default=4000)
    p.add_argument('--burn', type=int, default=1000)
    p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--seed', type=int, default=20261026)
    p.add_argument('--keep-traces', action='store_true')
    a = p.parse_args(argv)
    run_rep(a.study, a.rep, reps=a.reps, chains=a.chains,
            samples=a.samples, burn=a.burn, jobs=a.jobs,
            seed=a.seed, keep_traces=a.keep_traces)


if __name__ == '__main__':
    main()
