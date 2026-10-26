"""Generate and save trial-aligned posterior predictive summary draws for all six models.

Each run reads already-fitted HDDM chains; it never fits or modifies models.
500 replicated full datasets by default (100 from each of five chains),
generated in small batches to limit peak memory use.

- ES models: participant-level RT/accuracy by chosen E or S; P(E/S) by
  predicted RT quintile; P(S) by *observed* S-minus-E dwell-time quintile.
- EE accuracy model: overall accuracy and RT for correct/error choices.
- Tie trials are RETAINED in ES identity PPC; ties are excluded ONLY from
  accuracy calculations because correctness is undefined on a value tie.

PPC simulated rows are joined to their original trials via (sub_id, phase,
trial), then their appended empirical RT/response are verified. No positional
trial matching is accepted. This prevents false gaze/choice associations.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .models import ROOT, check_input, load_models

# Identical E/S palette to the original identity PPC; prediction is darker.
OBS_E = 'deepskyblue'
OBS_S = 'darkorchid'
PPC_E = 'steelblue'
PPC_S = 'indigo'
OBS_OTHER = '#B5B5B5'
PPC_OTHER = '#515151'
FONT_TITLE = 28
FONT_PANEL = 22
FONT_AXIS = 22
FONT_TICKS = 19
QUINTILES = (1, 2, 3, 4, 5)
IDENTITY_MODELS = frozenset({'addm', 'addm_z', 'esaddm', 'esaddm_z'})
TRIAL_KEYS = ['sub_id', 'phase', 'trial']


def model_prefix(model: str, study: str) -> str:
    return f'{model}_{study}'


def split_draws(total: int, chains: int) -> list[int]:
    if total < chains or chains < 1:
        raise ValueError('PPC replicates must be >= number of chains (both positive)')
    quotient, remainder = divmod(total, chains)
    return [quotient + (i < remainder) for i in range(chains)]


def split_batches(total: int, batch_size: int) -> list[int]:
    """Partition PPC replications without holding a full chain in memory."""
    if total < 1 or batch_size < 1:
        raise ValueError('PPC draw and batch counts must be positive')
    return [min(batch_size, total - start) for start in range(0, total, batch_size)]


def load_fit_metadata(study: str, model: str, chains: int, root: Path = ROOT):
    spec = load_models(root)[model]
    directory = root / 'studies' / study / 'results' / 'fits' / model
    prefix = model_prefix(model, study)
    manifest = directory / f'{prefix}_run.json'
    if not manifest.is_file():
        raise FileNotFoundError(f'Missing full-fit manifest: {manifest}')
    meta = json.loads(manifest.read_text(encoding='utf-8'))
    expected = {
        'study': study, 'model': model, 'formula': spec.drift_formula,
        'coordinate': spec.coordinate, 'included_ddm_parameters': spec.include,
        'chains': list(range(1, chains + 1)), 'status': 'completed',
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f'{manifest}: {key} is {meta.get(key)!r}, expected {value!r}')
    chain_files = [directory / f'{prefix}_chain{i}.hddm' for i in expected['chains']]
    missing = [str(p) for p in chain_files if not p.is_file()]
    if missing:
        raise FileNotFoundError(f'Missing full-fit HDDM models: {missing}')
    return chain_files, meta


def verify_fit_observed(model_data: pd.DataFrame, prepared: pd.DataFrame) -> pd.DataFrame:
    """Require exact row identities and RT/response agreement with fit inputs."""
    obs = model_data.copy().reset_index(drop=True)
    required = {*TRIAL_KEYS, 'subj_idx', 'rt', 'response'}
    if not required.issubset(obs) or not required.issubset(prepared):
        raise ValueError(f'Fit/prepared data missing {sorted(required - set(obs))}')
    if len(obs) != len(prepared):
        raise ValueError('Saved HDDM dataset and prepared dataset differ in trial count')
    if obs.duplicated(TRIAL_KEYS).any() or prepared.duplicated(TRIAL_KEYS).any():
        raise ValueError('Nonunique fitted trial identities')
    target = prepared[[*TRIAL_KEYS, 'subj_idx', 'rt', 'response']].rename(
        columns={c: f'__expected_{c}' for c in ('subj_idx', 'rt', 'response')})
    joined = obs.merge(target, on=TRIAL_KEYS, how='left', validate='one_to_one')
    if joined['__expected_rt'].isna().any():
        raise ValueError('Some fitted observations do not match prepared trials')
    for col in ('subj_idx', 'rt', 'response'):
        left = pd.to_numeric(joined[col]).abs() if col == 'rt' else pd.to_numeric(joined[col])
        right = pd.to_numeric(joined[f'__expected_{col}']).abs() if col == 'rt' else pd.to_numeric(joined[f'__expected_{col}'])
        if not np.allclose(left, right, atol=1e-5, rtol=0):
            raise ValueError(f'Saved HDDM observations disagree with prepared {col}')
    return obs


def attach_raw_es_metadata(obs: pd.DataFrame, study: str, root: Path = ROOT) -> pd.DataFrame:
    """Strict, one-to-one AnalysisReady merge; original corrected E/S gaze."""
    cfg = json.loads((root / 'config' / f'{study}.json').read_text(encoding='utf-8'))
    path = root / 'studies' / study / 'data' / 'raw' / cfg['raw_file']
    needed = [*TRIAL_KEYS, 'op1', 'op2', 'cho', 'p1', 'p2',
              'Dwell_S', 'Dwell_E', 'DwellTimeAdvantage_ES']
    raw = pd.read_csv(path, usecols=needed, low_memory=False)
    raw = raw.loc[raw.phase.astype(str).str.strip().eq('ES')].copy()
    raw['phase'] = raw.phase.astype(str).str.strip()
    if raw.duplicated(TRIAL_KEYS).any():
        raise ValueError('AnalysisReady ES trial keys are not unique')
    orig_n = len(obs)
    obs = obs.merge(raw, on=TRIAL_KEYS, how='left', validate='one_to_one',
                    suffixes=('', '_raw'), sort=False)
    if len(obs) != orig_n or obs[['op1', 'op2']].isna().any().any():
        raise ValueError('AnalysisReady E/S trial join failed')
    if not ((obs.op1.eq('E') & obs.op2.eq('S')) |
            (obs.op1.eq('S') & obs.op2.eq('E'))).all():
        raise ValueError('ES raw trials do not each contain E and S')
    for col in ['p1', 'p2', 'Dwell_S', 'Dwell_E', 'DwellTimeAdvantage_ES']:
        obs[col] = pd.to_numeric(obs[col], errors='raise')
    if obs[['Dwell_S', 'Dwell_E', 'DwellTimeAdvantage_ES']].isna().any().any():
        raise ValueError('Fitted ES rows have missing corrected gaze information')
    if not np.allclose(obs['DwellTimeAdvantage_ES'], obs['Dwell_S']-obs['Dwell_E'],
                       atol=1e-7, rtol=0):
        raise ValueError('DTA must equal Dwell_S minus Dwell_E, not spatial DTA')
    if 'cho_raw' in obs and obs['cho_raw'].notna().all():
        if not np.allclose(pd.to_numeric(obs.cho), pd.to_numeric(obs.cho_raw)):
            raise ValueError('Choice in raw and prepared data does not match')
    high_side = np.where(obs.p1 > obs.p2, 1, np.where(obs.p2 > obs.p1, 2, 0))
    obs['better_source'] = np.where(high_side == 1, obs.op1,
                                    np.where(high_side == 2, obs.op2, None))
    obs['worse_source'] = np.where(obs.better_source.eq('E'), 'S',
                                   np.where(obs.better_source.eq('S'), 'E', None))
    obs['is_tie'] = high_side == 0
    obs['dwell_ms_s_minus_e'] = obs['DwellTimeAdvantage_ES']
    if 'v_e' in obs and 'v_s' in obs:
        if not np.array_equal(obs['is_tie'].to_numpy(),
                              np.isclose(obs.v_e, obs.v_s, atol=1e-10, rtol=0)):
            raise ValueError('Raw/prepared ES tie status disagrees')
    return obs


def set_observed_outcomes(obs: pd.DataFrame, model: str) -> pd.DataFrame:
    obs = obs.copy()
    if model == 'accuracy_ee':
        obs['choice_group'] = np.where(obs.response.eq(1), 'Correct', 'Error')
        obs['accuracy_value'] = obs.response.astype(float)
        return obs
    if model == 'accuracy_es':
        obs['choice_group'] = np.where(obs.response.eq(1), obs.better_source,
                                        obs.worse_source)
        direct = np.where(pd.to_numeric(obs.cho).eq(1), obs.op1, obs.op2)
        if not np.array_equal(obs.choice_group.to_numpy(), direct):
            raise ValueError('Observed accuracy-coded choices disagree with E/S source labels')
        obs['accuracy_value'] = obs.response.astype(float)
    else:
        obs['choice_group'] = np.where(obs.response.eq(1), 'S', 'E')
        obs['accuracy_value'] = np.where(
            obs['is_tie'], np.nan,
            (obs['choice_group'] == obs['better_source']).astype(float))
        # Identity-coded E/S ties remain in choice and RT PPC, not accuracy.
    if not obs.choice_group.isin(['E', 'S']).all():
        raise ValueError('Missing E/S choice group')
    return obs


def rank_quintile(x: pd.Series) -> np.ndarray:
    """Five equally-sized, ascending RT or DTA bins; tie order is deterministic."""
    numeric = np.asarray(pd.to_numeric(x, errors='raise'), dtype=float)
    if len(numeric) < 5 or not np.isfinite(numeric).all():
        raise ValueError('Cannot rank-bin fewer than five finite observations')
    order = np.argsort(numeric, kind='stable')
    groups = np.empty(len(numeric), dtype=np.int8)
    for i, indices in enumerate(np.array_split(order, 5), start=1):
        groups[indices] = i
    return groups


def prepare_observed(study: str, model: str, prepared: pd.DataFrame,
                     saved_data: pd.DataFrame, root: Path = ROOT) -> pd.DataFrame:
    obs = verify_fit_observed(saved_data, prepared)
    if model != 'accuracy_ee':
        obs = attach_raw_es_metadata(obs, study, root)
    obs = set_observed_outcomes(obs, model)
    obs['rt_value'] = np.abs(pd.to_numeric(obs.rt, errors='raise'))
    obs['rt_quintile'] = rank_quintile(obs.rt_value)
    if model != 'accuracy_ee':
        obs['dwell_quintile'] = rank_quintile(obs.dwell_ms_s_minus_e)
    return obs


def safe_reset_index(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    names = []
    for idx, original in enumerate(frame.index.names):
        name = original if original is not None else f'index_level_{idx}'
        while name in frame.columns or name in names:
            name = f'index_{idx}_{name}'
        names.append(name)
    frame.index = frame.index.set_names(names)
    return frame.reset_index()


def align_predictions_by_observed_node(raw_pred: pd.DataFrame, hddm_model) -> pd.DataFrame:
    """Align regression PPC by the observed wfpt node and within-node row index.

    Kabuki appends empirical columns by DataFrame join when append_data=True.
    For regression simulations this can duplicate or misattach trial identifiers.
    Instead use the observed node's indexed fit rows and the third PPC index.
    Abort if their relationship is not unambiguous.
    """
    if not isinstance(raw_pred.index, pd.MultiIndex) or raw_pred.index.nlevels != 3:
        raise ValueError('Expected HDDM PPC index (node, sample, row)')

    # HDDMRegressor's observed wfpt node contains the indexed fitted RTs.
    # Retrieve corresponding metadata by index label, NOT via Kabuki's
    # append_data join, which may be misaligned with regression simulations.
    fields = [*TRIAL_KEYS, 'subj_idx', 'rt', 'response']
    model_data = hddm_model.data
    if not model_data.index.is_unique:
        raise ValueError('Fitted HDDM dataframe index is nonunique')
    node_tables = {}
    for name, observed in hddm_model.iter_observeds():
        node_index = observed['node'].value.index
        if not node_index.is_unique:
            raise ValueError(f'HDDM node {name} has a nonunique observation index')
        table = model_data.loc[node_index, fields].copy()
        if len(table) != len(node_index) or not table.index.equals(node_index):
            raise ValueError(f'HDDM node {name} index does not align with saved model data')
        node_tables[str(name)] = table

    if not node_tables:
        raise ValueError('HDDM has no observed likelihood nodes')
    stacked = pd.concat(node_tables.values(), ignore_index=True)
    if len(stacked) != len(model_data) or stacked.duplicated(TRIAL_KEYS).any():
        raise ValueError('HDDM likelihood nodes do not partition fitted trials uniquely')

    flat = safe_reset_index(raw_pred)
    node_col, sample_col, row_col = list(flat.columns[:3])
    if 'rt' not in flat or 'response' not in flat:
        raise ValueError('Posterior predictive sample lacks simulated RT/response')
    if set(flat[node_col].astype(str)) != set(node_tables):
        raise ValueError('PPC node names do not agree with fitted observed nodes')

    parts = []
    for node_name, rows in flat.groupby(node_col, sort=False):
        table = node_tables[str(node_name)]
        row_ids = pd.Index(rows[row_col].unique())
        n = len(table)
        if len(row_ids) != n or rows.groupby(sample_col).size().ne(n).any():
            raise ValueError(f'PPC node {node_name} does not contain exactly {n} observations per draw')
        if not rows.duplicated([sample_col, row_col]).sum() == 0:
            raise ValueError(f'PPC node {node_name} generated repeated row positions in a draw')

        # Some HDDM simulators use original row labels, others reset each
        # observed node to positions 0..n-1. Both mappings are checked.
        if row_ids.isin(table.index).all() and len(row_ids) == len(table.index):
            mapped = table.loc[rows[row_col].to_numpy()].reset_index(drop=True)
        else:
            positions = pd.to_numeric(rows[row_col], errors='coerce').to_numpy()
            if (not np.isfinite(positions).all()
                    or not np.equal(positions, np.floor(positions)).all()
                    or not np.isin(positions, np.arange(n)).all()):
                raise ValueError(f'Unrecognised PPC row-index convention for node {node_name}')
            mapped = table.iloc[positions.astype(int)].reset_index(drop=True)

        part = rows[[node_col, sample_col]].reset_index(drop=True).copy()
        part['rt_sampled'] = rows['rt'].to_numpy()
        part['response_sampled'] = rows['response'].to_numpy()
        for field in fields:
            part[field] = mapped[field].to_numpy()
        parts.append(part)

    result = pd.concat(parts, ignore_index=True)
    if result.duplicated([sample_col, *TRIAL_KEYS]).any():
        raise ValueError('PPC node-aligned trial identities still duplicate; refusing to misassign gaze')
    return result.rename(columns={sample_col: 'sample'})


def identify_sample_column(flat: pd.DataFrame) -> str:
    for col in ('sample', 'draw', 'idx_sample'):
        if col in flat:
            return col
    matches = [c for c in flat if re.match(r'^(?:index_\d+_)?(?:sample|draw)$', str(c))]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f'Cannot locate HDDM PPC draw identifier; columns={list(flat)}')


def attach_predictive_outcomes(flat: pd.DataFrame, obs: pd.DataFrame,
                               model: str, chain: int) -> pd.DataFrame:
    """Reattach metadata by verified trial keys; NEVER infer by row position."""
    sample_col = identify_sample_column(flat)
    required = {*TRIAL_KEYS, 'subj_idx', 'rt', 'response',
                'response_sampled', 'rt_sampled', sample_col}
    if not required.issubset(flat.columns):
        raise ValueError(f'HDDM append_data PPC missing columns: {sorted(required-set(flat.columns))}')
    flat = flat.copy()
    flat['ppc_draw'] = flat[sample_col].astype(str).map(lambda x: f'{chain}:{x}')
    expected_n = len(obs)
    sizes = flat.groupby('ppc_draw', sort=False).size()
    if sizes.empty or not (sizes == expected_n).all():
        raise ValueError(f'Each PPC draw must contain exactly {expected_n} fitted rows: {sizes.describe()}')
    if flat.duplicated(['ppc_draw', *TRIAL_KEYS]).any():
        raise ValueError('Duplicate trial identity within PPC draw')
    # Verify the empirical trial identity against prepared data after reconstruction
    # from the observed HDDM nodes, never from Kabuki appended metadata.
    metadata = obs[[*TRIAL_KEYS, 'subj_idx', 'rt', 'response', 'better_source',
                    'worse_source', 'is_tie', 'dwell_quintile']].copy() if model != 'accuracy_ee' else obs[
                    [*TRIAL_KEYS, 'subj_idx', 'rt', 'response']].copy()
    metadata = metadata.rename(columns={k: f'__original_{k}'
                                        for k in ['subj_idx', 'rt', 'response']})
    flat = flat.merge(metadata, on=TRIAL_KEYS, how='left', validate='many_to_one', sort=False)
    if flat['__original_rt'].isna().any():
        raise ValueError('PPC trial keys fail to match fitted trial keys')
    for name in ('subj_idx', 'rt', 'response'):
        original = np.asarray(pd.to_numeric(flat[f'__original_{name}']), dtype=float)
        appended = np.asarray(pd.to_numeric(flat[name]), dtype=float)
        if name == 'rt':
            appended, original = np.abs(appended), np.abs(original)
        if not np.allclose(appended, original, rtol=0, atol=1e-5):
            raise ValueError(f'HDDM PPC appended {name} differs from fitted trial: alignment invalid')
    resp = pd.to_numeric(flat.response_sampled, errors='raise').to_numpy(dtype=float)
    if not np.isfinite(resp).all() or not np.isin(resp, [0, 1]).all():
        raise ValueError('HDDM simulated boundary responses must be 0/1')
    rt = np.abs(pd.to_numeric(flat.rt_sampled, errors='raise').to_numpy(dtype=float))
    if not np.isfinite(rt).all() or (rt <= 0).any():
        raise ValueError('PPC returned invalid simulated RTs')
    flat['rt_value'] = rt
    flat['rt_quintile'] = -1
    for _, indices in flat.groupby('ppc_draw', sort=False).groups.items():
        flat.loc[indices, 'rt_quintile'] = rank_quintile(flat.loc[indices, 'rt_value'])
    if model == 'accuracy_ee':
        flat['choice_group'] = np.where(resp == 1, 'Correct', 'Error')
        flat['accuracy_value'] = resp
    elif model == 'accuracy_es':
        flat['choice_group'] = np.where(resp == 1, flat.better_source, flat.worse_source)
        flat['accuracy_value'] = resp
    else:
        flat['choice_group'] = np.where(resp == 1, 'S', 'E')
        flat['accuracy_value'] = np.where(
            flat.is_tie, np.nan,
            (flat.choice_group == flat.better_source).astype(float))
    return flat


def subjectwise_mean(frame: pd.DataFrame, field: str, group: str | None = None) -> pd.DataFrame:
    keys = ['subj_idx'] + ([group] if group else [])
    return frame.groupby(keys, observed=True, sort=False)[field].mean().reset_index(name='value')


def bin_means(frame: pd.DataFrame, bin_name: str, field: str) -> pd.DataFrame:
    sub = frame.groupby(['subj_idx', bin_name], sort=False, observed=True)[field].mean().reset_index()
    return sub.groupby(bin_name, sort=True, observed=True)[field].mean().reindex(QUINTILES).reset_index(name='value')


def draw_metrics(frame: pd.DataFrame, study: str, model: str, draw: str) -> tuple[list[dict], list[dict]]:
    """Compact metrics for one simulated full dataset, with equal subject weighting."""
    out, subjects = [], []
    def push(metric: str, group: str, value: float, quintile=None):
        if not np.isfinite(value):
            raise ValueError(f'No finite estimate of {metric} {group} for draw {draw}')
        out.append(dict(study=study, model=model, draw=draw,
                        metric=metric, group=group, quintile=quintile, value=float(value)))

    def save_subjects(metric: str, group_col: str | None, valid=None):
        data = frame if valid is None else frame.loc[valid]
        sub = subjectwise_mean(data, metric, group_col)
        metric_label = {'rt_value': 'mean_rt', 'accuracy_value': 'accuracy'}[metric]
        for row in sub.itertuples(index=False):
            if group_col is None:
                subject, value = row
                group = 'Overall'
            else:
                subject, group, value = row
            if pd.notna(value):
                subjects.append(dict(draw=draw, subj_idx=int(subject),
                                     metric=metric_label, group=str(group), value=float(value)))
        if group_col is None:
            push(metric_label, 'Overall', sub.value.mean())
        else:
            for group, group_rows in sub.groupby(group_col, observed=True):
                push(metric_label, str(group), group_rows.value.mean())

    save_subjects('rt_value', None)
    save_subjects('accuracy_value', None)
    if model == 'accuracy_ee':
        save_subjects('rt_value', 'choice_group')
    else:
        save_subjects('rt_value', 'choice_group')
        save_subjects('accuracy_value', 'choice_group')
    if model in IDENTITY_MODELS:
        push('p_choose_s', 'S', subjectwise_mean(
            frame.assign(__chose_s=frame.choice_group.eq('S').astype(float)), '__chose_s').value.mean())
        data = frame.assign(__chose_s=frame.choice_group.eq('S').astype(float))
        rt_bins = bin_means(data, 'rt_quintile', '__chose_s')
        for b in QUINTILES:
            push('p_choose_s_by_rt_quintile', 'S',
                 rt_bins.loc[rt_bins.rt_quintile.eq(b), 'value'].iloc[0], b)
        dwell = bin_means(data, 'dwell_quintile', '__chose_s')
        for b in QUINTILES:
            push('p_choose_s_by_dwell_quintile', 'S',
                 dwell.loc[dwell.dwell_quintile.eq(b), 'value'].iloc[0], b)
        push('p_choose_s_rt_quintile_slope', 'S',
             np.polyfit(np.array(QUINTILES), rt_bins.value.to_numpy(float), 1)[0])
        push('p_choose_s_dwell_quintile_slope', 'S',
             np.polyfit(np.array(QUINTILES), dwell.value.to_numpy(float), 1)[0])
    return out, subjects


def observed_metrics(obs: pd.DataFrame, study: str, model: str):
    return draw_metrics(obs, study, model, 'observed')


def summarise_metrics(obs_rows: pd.DataFrame, sim_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ['metric', 'group', 'quintile']
    obs = obs_rows.copy()
    sim = sim_rows.copy()
    obs['quintile'] = pd.to_numeric(obs['quintile']).fillna(0).astype(int)
    sim['quintile'] = pd.to_numeric(sim['quintile']).fillna(0).astype(int)
    g = sim.groupby(keys, dropna=False, sort=False)['value']
    pred = g.agg(ppc_mean='mean', ppc_sd='std', ppc_median='median', n_ppc='size').reset_index()
    pi = g.quantile([.025, .975]).unstack().rename(columns={.025:'ppc_pi95_low', .975:'ppc_pi95_high'}).reset_index()
    pred = pred.merge(pi, on=keys, validate='one_to_one')
    joined = obs[keys + ['value']].rename(columns={'value':'empirical'}).merge(
        pred, on=keys, how='outer', validate='one_to_one')
    if joined.isna().any().any():
        raise ValueError('Metrics missing empirical values or predictive draws')
    joined['ppc_minus_empirical'] = joined.ppc_mean-joined.empirical
    joined['absolute_error'] = joined.ppc_minus_empirical.abs()
    joined['empirical_in_ppc_95'] = joined.empirical.between(joined.ppc_pi95_low, joined.ppc_pi95_high)
    core = joined[joined.quintile.eq(0)].drop(columns='quintile').reset_index(drop=True)
    curves = joined[joined.quintile.ne(0)].reset_index(drop=True)
    return core, curves


def subject_boxplot_data(obs_subjects: pd.DataFrame, sim_subjects: pd.DataFrame,
                         model: str, study: str) -> pd.DataFrame:
    keys = ['subj_idx','metric','group']
    obs = obs_subjects[keys + ['value']].rename(columns={'value':'empirical_value'})
    sim_group = sim_subjects.groupby(keys)['value']
    sim = sim_group.agg(ppc_subject_mean='mean', ppc_subject_sd='std', n_ppc='size').reset_index()
    pi = sim_group.quantile([.025, .975]).unstack().rename(
        columns={.025: 'ppc_subject_pi95_low', .975: 'ppc_subject_pi95_high'}).reset_index()
    sim = sim.merge(pi, on=keys, validate='one_to_one')
    out = obs.merge(sim, on=keys, how='left', validate='one_to_one')
    if out[['empirical_value','ppc_subject_mean']].isna().any().any():
        raise ValueError('A subject has no posterior-predictive summary for a plotted outcome')
    out.insert(0,'study',study)
    out.insert(1,'model',model)
    return out


def run_ppc(study: str, model: str, *, chains: int = 5, ppc_draws: int = 500,
            batch_size: int = 20, seed: int = 20260919, root: Path = ROOT) -> Path:
    import hddm  # Deferred: unit tests and CLI validation do not require HDDM.

    spec = load_models(root)[model]
    files, fit_meta = load_fit_metadata(study, model, chains, root)
    prepared, _ = check_input(study, spec, root)
    prefix = model_prefix(model, study)
    output = root / 'studies' / study / 'results' / 'ppc' / model
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'PPC output already exists; archive it before rerunning: {output}')
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f'{prefix}_ppc_run.json'
    allocated = split_draws(ppc_draws, chains)
    split_batches(max(allocated), batch_size)  # Validate before creating the fit outputs.
    rng = np.random.default_rng(seed)
    # Set RNG just before each post_pred_gen call: HDDM uses NumPy global random state.
    details = {'study':study, 'model':model, 'fit_manifest':str(files[0].parent / f'{prefix}_run.json'),
               'status':'running', 'chains':chains, 'ppc_draws_requested':ppc_draws, 'batch_size':batch_size,
               'draws_per_chain':allocated, 'seed':seed, 'started_utc':datetime.now(timezone.utc).isoformat(),
               'ppc_method':'HDDM post_pred_gen, append_data=False; verified per-node trial alignment',
               'rt_quintile':'independent rank quintiles in each predictive dataset',
               'dwell_quintile':'empirical corrected Dwell_S - Dwell_E rank quintile, fixed across replicates',
               'group_average':'equal weighting of participant-level means',
               'tie_handling':'identity ES ties retained except undefined accuracy',
               'boxplot_ppc':'distribution of subject-level predictive means across posterior replicates',
               'output_stage':'generation only: tables and figures built with esaddm.ppc_plot'}
    manifest_path.write_text(json.dumps(details,indent=2)+'\n',encoding='utf-8')
    observed = None
    observed_metrics_rows = None
    observed_subjects = None
    draw_rows, subject_rows = [], []
    # Save raw simulated RT/choice with minimal columns, streamed per chain.
    # This permits additional PPC analyses later without repeating HDDM simulation.
    trial_dir = output / 'data' / 'trials'
    trial_dir.mkdir(parents=True, exist_ok=True)
    trial_archives = []
    for ci, (path, count) in enumerate(zip(files, allocated), start=1):
        print(f'LOADING {model}/{study} chain {ci}/{chains}: {path}', flush=True)
        hddm_model = hddm.load(str(path))
        if observed is None:
            observed = prepare_observed(study, model, prepared, hddm_model.data, root)
            observed_metrics_rows, observed_subjects = observed_metrics(observed, study, model)
            obs_trials = trial_dir / f'{prefix}_observed_trials.csv.gz'
            observed.to_csv(obs_trials, index=False, compression='gzip')
            trial_archives.append(str(obs_trials.relative_to(output)))
        else:
            # Compare every chain's fitted data with the same prepared input.
            verify_fit_observed(hddm_model.data, prepared)
        np.random.seed(int(rng.integers(1, 2**31 - 1)))
        print(f'PREDICT {model}/{study} chain {ci}: {count} full trial datasets', flush=True)
        generated = 0
        sim_trial_file = trial_dir / f'{prefix}_simulated_chain{ci}.csv.gz'
        trial_archives.append(str(sim_trial_file.relative_to(output)))
        for batch_i, batch_count in enumerate(split_batches(count, batch_size), start=1):
            # Generate/aggregate one small batch, then immediately discard trial-level data.
            raw_pred = hddm.utils.post_pred_gen(
                hddm_model, samples=batch_count, append_data=False)
            flat = align_predictions_by_observed_node(raw_pred, hddm_model)
            del raw_pred
            predicted = attach_predictive_outcomes(flat, observed, model, ci)
            del flat
            # post_pred_gen restarts draw numbering on every call: namespace each batch.
            predicted['ppc_draw'] = (f'{ci}:batch{batch_i}:' +
                                     predicted['ppc_draw'].astype(str))
            actual = predicted.ppc_draw.nunique()
            if actual != batch_count:
                raise RuntimeError(f'Chain {ci} batch {batch_i}: expected '
                                   f'{batch_count} datasets, got {actual}')
            for draw, block in predicted.groupby('ppc_draw', sort=False):
                rows, subjects = draw_metrics(block, study, model, draw)
                draw_rows.extend(rows)
                subject_rows.extend(subjects)
            # Write each batch immediately: no accumulation of 500 trial-level datasets.
            trial_cols = ['ppc_draw', *TRIAL_KEYS, 'subj_idx', 'rt_value', 'response_sampled']
            predicted[trial_cols].to_csv(
                sim_trial_file, index=False, mode='wt' if batch_i == 1 else 'at',
                header=(batch_i == 1), compression='gzip', float_format='%.10g')
            generated += actual
            print(f'PREDICT {model}/{study} chain {ci}: '
                  f'{generated}/{count} datasets summarised', flush=True)
            del predicted
            gc.collect()
        if generated != count:
            raise RuntimeError(f'Chain {ci}: generated {generated}, expected {count}')
        print(f'PASS chain {ci}: {generated} aligned replicates', flush=True)
        del hddm_model
        gc.collect()

    # Persist the sufficient summaries *before* any plotting/tables. The PPC
    # simulation must be run only once; plots and presentation tables can be
    # rebuilt independently from these saved per-draw/per-subject metrics.
    sim_metrics = pd.DataFrame(draw_rows)
    sim_subjects = pd.DataFrame(subject_rows)
    empirical_metrics = pd.DataFrame(observed_metrics_rows)
    empirical_subjects = pd.DataFrame(observed_subjects)
    n_generated = sim_metrics['draw'].nunique()
    if n_generated != ppc_draws or sim_subjects['draw'].nunique() != ppc_draws:
        raise ValueError(f'Only {n_generated} predictive datasets produced; requested {ppc_draws}')

    data_dir = output / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    saved = {
        'observed_metrics': empirical_metrics,
        'observed_subject_metrics': empirical_subjects,
        'replicate_metrics': sim_metrics,
        'replicate_subject_metrics': sim_subjects,
    }
    for label, frame in saved.items():
        target = data_dir / f'{prefix}_{label}.csv'
        temporary = target.with_suffix('.csv.part')
        frame.to_csv(temporary, index=False, float_format='%.10g')
        temporary.replace(target)
        print(f'SAVED PPC DATA: {target} ({len(frame)} rows)', flush=True)

    details.update({
        'ppc_draws_generated': int(n_generated),
        'rows_in_fitted_data': int(len(observed)),
        'participants': int(observed.subj_idx.nunique()),
        'saved_data_files': [str((data_dir / f'{prefix}_{label}.csv').relative_to(output))
                             for label in saved],
        'saved_trial_archives': trial_archives,
        'status': 'completed',
        'finished_utc': datetime.now(timezone.utc).isoformat(),
    })
    manifest_path.write_text(json.dumps(details, indent=2) + '\n', encoding='utf-8')
    print(f'PASS PPC GENERATION {study}/{model}: {n_generated} aligned replicated '
          f'datasets saved; no tables or figures generated', flush=True)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description='Generate and persist PPC simulations, without making figures or summary tables')
    parser.add_argument('--study', choices=['study1','study2'], required=True)
    parser.add_argument('--model', choices=[*load_models(), 'all'], required=True)
    parser.add_argument('--chains',type=int,default=5)
    parser.add_argument('--draws',type=int,default=500,
                        help='Number of full replicated trial datasets pooled across chains')
    parser.add_argument('--batch-size',type=int,default=20,
                        help='Replicated datasets per post_pred_gen call (limits peak RAM)')
    parser.add_argument('--seed',type=int,default=20260919)
    args=parser.parse_args(argv)
    models=list(load_models()) if args.model=='all' else [args.model]
    for model in models:
        run_ppc(args.study,model,chains=args.chains,ppc_draws=args.draws,batch_size=args.batch_size,seed=args.seed)

if __name__ == '__main__':
    main()
