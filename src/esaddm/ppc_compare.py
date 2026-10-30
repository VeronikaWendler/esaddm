"""Four-model ES posterior predictive comparisons from saved PPC data only.

Compares aDDM, aDDM+z, ESaDDM, ESaDDM+z within ONE study.
Does not import HDDM, simulate decisions, or alter existing PPC outputs.

All four models must use exactly matching observed identity-coded ES trials.
Median RT is calculated once from stored compressed trial-level PPC outputs
and cached with a source-file fingerprint for fast subsequent replotting.
All group summaries weight participants equally, as in esaddm.ppc.py.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd

from .models import ROOT
from .ppc import model_prefix, summarise_metrics
from .ppc_plot import rstyle_curves_from_archives, _merge_rstyle_curves

MODELS = ('addm', 'addm_z', 'esaddm', 'esaddm_z')
LABELS = {
    'addm': 'aDDM', 'addm_z': 'aDDM + z',
    'esaddm': 'ESaDDM', 'esaddm_z': 'ESaDDM + z',
}
COLORS = {
    'addm': '#1f77b4', 'addm_z': '#ff7f0e',
    'esaddm': '#2ca02c', 'esaddm_z': '#d62728',
}
OBS_COLOR = '#222222'
EMP_E = '#00BFFF'
EMP_S = '#9932CC'
EDGE = '#000000'
RT_LABELS = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
DWELL_LABELS = ['E>>S', 'E>S', 'E~S', 'S>E', 'S>>E']

FONT_TITLE = 18
FONT_AXIS = 18
FONT_TICKS = 18
FONT_LEGEND = 16
FIG_WIDTH = 5.5
FIG_HEIGHT = 4.8

METRICS = (
    ('choice_probability', None,
     'Overall choice probability',
     'Choice', 'Choice probability (%)', ('E', 'S'),
     'overall_choice_probability_by_response_category'),
    ('mean_rt', None,
     'Mean RT conditional on choice',
     'Choice', 'Mean RT (s)', ('E', 'S'),
     'mean_rt_conditional_on_choice'),
    ('median_rt', None,
     'Median RT conditional on choice',
     'Choice', 'Median RT (s)', ('E', 'S'),
     'median_rt_conditional_on_choice'),
    ('p_choose_s_by_rt_quintile', 'S',
     'Choice by RT quintile',
     'RT quintile (fast to slow)', 'Probability (%)',
     (1, 2, 3, 4, 5), 'p_choose_s_by_rt_quintile'),
    ('p_choose_s_by_dwell_quintile', 'S',
     'Choice by dwell-time advantage',
     'Dwell advantage (S − E)', 'P(Choose Symbolic)',
     (1, 2, 3, 4, 5), 'p_choose_s_by_dwell_advantage_quintile'),
    ('accuracy', None,
     'Choice accuracy by category',
     'Choice', 'Accuracy (%)', ('E', 'S'),
     'accuracy_by_response_category'),
)

CACHE_VERSION = 2
TRIAL_KEY = ('sub_id', 'phase', 'trial')
AUDIT_COLUMNS = ('sub_id', 'phase', 'trial', 'subj_idx', 'response', 'rt',
                 'rt_value', 'choice_group', 'accuracy_value', 'is_tie',
                 'better_source', 'dwell_ms_s_minus_e', 'dwell_quintile')


def _read(path: Path, **kwargs) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f'Missing completed PPC input: {path}')
    frame = pd.read_csv(path, **kwargs)
    if frame.empty:
        raise ValueError(f'Empty PPC input: {path}')
    return frame


def _atomic_csv(frame: pd.DataFrame, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_name(dest.name + '.part')
    frame.to_csv(temp, index=False, float_format='%.10g')
    temp.replace(dest)


def _style_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.6)
    ax.spines['bottom'].set_linewidth(1.6)
    ax.tick_params(axis='both', labelsize=FONT_TICKS, width=1.3)
    ax.grid(False)


def _percent_formatter(x, pos):
    return f'{int(round(x))}%'


def _set_limits(ax, values, is_probability=False, include_half=False, percent=False):
    arr = np.concatenate([np.asarray(v, dtype=float).ravel() for v in values])
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return
    if percent:
        lo = max(0.0, np.floor((arr.min() - 3.0) / 5.0) * 5.0)
        hi = min(100.0, np.ceil((arr.max() + 3.0) / 5.0) * 5.0)
        if include_half:
            lo = min(lo, 45.0)
            hi = max(hi, 55.0)
        ax.set_ylim(lo, hi)
        ax.yaxis.set_major_formatter(FuncFormatter(_percent_formatter))
        return
    if is_probability:
        lo = max(0.0, arr.min() - 0.04)
        hi = min(1.0, arr.max() + 0.04)
        if include_half:
            lo = min(lo, 0.45)
            hi = max(hi, 0.55)
        ax.set_ylim(lo, hi)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        return
    vmin, vmax = float(arr.min()), float(arr.max())
    span = max(vmax - vmin, 0.01)
    pad = span * 0.12
    ax.set_ylim(vmin - pad, vmax + pad)


def _verify_observed(reference: pd.DataFrame, current: pd.DataFrame, model: str):
    cols = [c for c in AUDIT_COLUMNS if c in reference.columns and c in current.columns]
    missing = sorted(set(AUDIT_COLUMNS) - set(reference.columns).intersection(current.columns))
    if missing:
        raise ValueError(f'Observed trial records lack audit columns {missing} ({model})')

    def canon(frame):
        if frame.duplicated(list(TRIAL_KEY)).any():
            raise ValueError('Duplicate empirical trial identity')
        if not frame['phase'].astype(str).eq('ES').all():
            raise ValueError('Identity-coded comparison requires ES trials only')
        if not frame['response'].isin((0, 1)).all():
            raise ValueError('Identity-coded response must be binary')
        if not np.array_equal(frame['choice_group'].to_numpy(),
                              np.where(frame['response'].eq(1), 'S', 'E')):
            raise ValueError('Choice coding disagrees with S-upper/E-lower identity')
        return frame.sort_values(list(TRIAL_KEY)).reset_index(drop=True)

    a, b = canon(reference), canon(current)
    if len(a) != len(b):
        raise ValueError(f'Fitted trial counts differ across models ({model})')
    for col in cols:
        left, right = a[col], b[col]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            ok = np.allclose(left.to_numpy(dtype=float), right.to_numpy(dtype=float),
                             atol=1e-7, rtol=0, equal_nan=True)
        else:
            ok = left.fillna('<MISSING>').astype(str).equals(
                right.fillna('<MISSING>').astype(str))
        if not ok:
            raise ValueError(f'Empirical data mismatch across models: {model}/{col}')


def load_sources(study: str, root: Path):
    result_root = root / 'studies' / study / 'results' / 'ppc'
    sources, audit = {}, []
    reference_trials = None
    reference_observed = None
    n_draws_ref = None
    for model in MODELS:
        directory = result_root / model
        prefix = model_prefix(model, study)
        meta_file = directory / f'{prefix}_ppc_run.json'
        if not meta_file.exists():
            raise FileNotFoundError(f'No PPC generation manifest: {meta_file}')
        meta = json.loads(meta_file.read_text(encoding='utf-8'))
        if any((meta.get('status') != 'completed', meta.get('study') != study, meta.get('model') != model)):
            raise ValueError(f'Incomplete or mismatched PPC generation: {meta_file}')
        n_draws = meta.get('ppc_draws_generated')
        if not isinstance(n_draws, int) or n_draws < 2:
            raise ValueError(f'Invalid PPC replication count: {meta_file}')
        if n_draws_ref is None:
            n_draws_ref = n_draws
        elif n_draws != n_draws_ref:
            raise ValueError(f'Different PPC draw counts: {n_draws_ref} versus {n_draws} ({model})')

        rawdir = directory / 'data'
        observed = _read(rawdir / f'{prefix}_observed_metrics.csv')
        simulated = _read(rawdir / f'{prefix}_replicate_metrics.csv')
        if simulated['draw'].nunique() != n_draws:
            raise ValueError(f'Saved draw count disagrees with manifest: {model}')
        if observed.duplicated(['metric', 'group', 'quintile']).any():
            raise ValueError(f'Duplicate observed summary keys: {model}')
        if simulated.duplicated(['draw', 'metric', 'group', 'quintile']).any():
            raise ValueError(f'Duplicate predictive summary keys: {model}')
        if not observed['model'].eq(model).all() or not simulated['model'].eq(model).all():
            raise ValueError(f'Incorrect model label in saved metrics: {model}')

        core, curves = summarise_metrics(observed, simulated)
        curves = _merge_rstyle_curves(curves, rstyle_curves_from_archives(
            study, model, directory, meta))
        trials = _read(rawdir / 'trials' / f'{prefix}_observed_trials.csv.gz')
        if reference_trials is None:
            reference_trials = trials
            _verify_observed(trials, trials, model)
        else:
            _verify_observed(reference_trials, trials, model)
        empirical = pd.concat([core, curves], ignore_index=True)
        empirical = empirical[['metric', 'group', 'quintile', 'empirical']].copy()
        empirical['quintile'] = empirical.quintile.fillna(0).astype(int)
        empirical = empirical.sort_values(['metric', 'group', 'quintile']).reset_index(drop=True)
        if reference_observed is None:
            reference_observed = empirical
        else:
            if not reference_observed[['metric', 'group', 'quintile']].equals(
                empirical[['metric', 'group', 'quintile']]) or not np.allclose(
                reference_observed.empirical, empirical.empirical, atol=1e-7, rtol=0):
                raise ValueError(f'Observed PPC metric summaries differ across models: {model}')
        sources[model] = dict(
            directory=directory, prefix=prefix, manifest=meta,
            observed=observed, simulated=simulated, core=core, curves=curves, trials=trials
        )
        audit.append(dict(study=study, model=model, model_label=LABELS[model],
                          observed_trials=len(trials), subjects=trials.subj_idx.nunique(),
                          ppc_draws=n_draws, status='passed'))
    return sources, pd.DataFrame(audit)


def _median_from_observed(trials: pd.DataFrame, study: str, model: str) -> pd.DataFrame:
    df = trials[['subj_idx', 'choice_group', 'rt_value']].copy()
    df['rt_value'] = np.abs(pd.to_numeric(df['rt_value'], errors='raise'))
    if df.rt_value.isna().any() or (df.rt_value <= 0).any():
        raise ValueError('Invalid empirical RT for median calculation')
    overall = df.groupby('subj_idx')['rt_value'].median().mean()
    grouped = df.groupby(['subj_idx', 'choice_group'])['rt_value'].median().groupby('choice_group').mean()
    if not set(('E', 'S')).issubset(grouped.index):
        raise ValueError('Missing source choice in observed median RT')
    return pd.DataFrame([
        dict(study=study, model=model, draw='observed', metric='median_rt', group=g, value=float(v))
        for g, v in [('Overall', overall), ('E', grouped['E']), ('S', grouped['S'])]
    ])


def _median_from_simulated(path: Path, study: str, model: str,
                           n_obs_trials: int, n_expected_draws: int) -> pd.DataFrame:
    frame = _read(path, usecols=['ppc_draw', 'subj_idx', 'rt_value', 'response_sampled'],
                  dtype={'ppc_draw': 'str', 'subj_idx': 'int32',
                         'rt_value': 'float64', 'response_sampled': 'int8'})
    count = frame.groupby('ppc_draw', observed=True).size()
    if len(count) != n_expected_draws or not count.eq(n_obs_trials).all():
        raise ValueError(f'Incorrect number of predicted trials or PPC draws: {path}')
    if not frame.response_sampled.isin([0, 1]).all():
        raise ValueError(f'Invalid simulated choice in {path}')
    if not np.isfinite(frame.rt_value.to_numpy()).all() or (frame.rt_value <= 0).any():
        raise ValueError(f'Invalid simulated RT in {path}')
    frame['choice_group'] = np.where(frame.response_sampled.eq(1), 'S', 'E')
    overall = (frame.groupby(['ppc_draw', 'subj_idx'], observed=True).rt_value.median()
               .groupby('ppc_draw', observed=True).mean())
    by_source = (frame.groupby(['ppc_draw', 'subj_idx', 'choice_group'], observed=True)
                 .rt_value.median().groupby(['ppc_draw', 'choice_group'], observed=True).mean())
    output = []
    for draw, val in overall.items():
        output.append(dict(study=study, model=model, draw=draw,
                           metric='median_rt', group='Overall', value=float(val)))
    for (draw, group), val in by_source.items():
        output.append(dict(study=study, model=model, draw=draw,
                           metric='median_rt', group=group, value=float(val)))
    medians = pd.DataFrame(output)
    counts = medians.groupby('draw').group.nunique()
    if len(counts) != n_expected_draws or not counts.eq(3).all():
        raise ValueError(f'Missing E or S median RT for some draws: {path}')
    return medians


def _median_inputs_signature(study: str, sources: dict) -> dict:
    def fingerprint(path):
        st = path.stat()
        return [str(path.resolve()), st.st_size, st.st_mtime_ns]
    files = []
    for model, source in sources.items():
        trial_dir = source['directory'] / 'data' / 'trials'
        prefix = source['prefix']
        files.append(fingerprint(trial_dir / f'{prefix}_observed_trials.csv.gz'))
        n_chains = source['manifest'].get('chains', 5)
        if not isinstance(n_chains, int) or n_chains < 1:
            raise ValueError(f'Invalid PPC chain count: {model}')
        for chain in range(1, n_chains + 1):
            files.append(fingerprint(trial_dir / f'{prefix}_simulated_chain{chain}.csv.gz'))
    return dict(version=CACHE_VERSION, study=study, files=files)


def load_or_calculate_medians(study: str, sources: dict, outdir: Path,
                              rebuild: bool = False) -> pd.DataFrame:
    cache_dir = outdir / 'data'
    cache_file = cache_dir / f'{study}_median_rt_by_draw.csv'
    meta_file = cache_dir / f'{study}_median_rt_cache.json'
    signature = _median_inputs_signature(study, sources)
    if not rebuild and cache_file.is_file() and meta_file.is_file():
        if json.loads(meta_file.read_text(encoding='utf-8')) == signature:
            cached = _read(cache_file, dtype={'draw': str})
            print(f'REUSING verified median RT cache: {cache_file}', flush=True)
            return cached
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for model, source in sources.items():
        prefix = source['prefix']
        rows.append(_median_from_observed(source['trials'], study, model))
        n_chains = source['manifest'].get('chains', 5)
        allocations = source['manifest'].get('draws_per_chain')
        if allocations is None:
            raise ValueError(f'Missing PPC draw allocation in manifest for {model}')
        if len(allocations) != n_chains or sum(allocations) != source['manifest']['ppc_draws_generated']:
            raise ValueError(f'Bad PPC chain allocations for {model}')
        for chain, expected in enumerate(allocations, 1):
            path = source['directory'] / 'data' / 'trials' / f'{prefix}_simulated_chain{chain}.csv.gz'
            print(f'MEDIAN RT from saved simulations: {study}/{model} chain {chain}/{n_chains}', flush=True)
            rows.append(_median_from_simulated(path, study, model, len(source['trials']), int(expected)))
    cache = pd.concat(rows, ignore_index=True)
    _atomic_csv(cache, cache_file)
    temporary = meta_file.with_name(meta_file.name + '.part')
    temporary.write_text(json.dumps(signature, indent=2) + '\n', encoding='utf-8')
    temporary.replace(meta_file)
    return cache


def median_summary(cache: pd.DataFrame, model: str) -> pd.DataFrame:
    selected = cache.loc[cache.model.eq(model)].copy()
    obs = selected.loc[selected.draw.eq('observed')].copy()
    pred = selected.loc[~selected.draw.eq('observed')].copy()
    obs['quintile'] = np.nan
    pred['quintile'] = np.nan
    core, curves = summarise_metrics(obs, pred)
    if not curves.empty:
        raise ValueError('Median RT should not have quintile curves')
    return core


def _choice_probability_rows(source_core: pd.DataFrame) -> pd.DataFrame:
    s = source_core.loc[source_core.metric.eq('p_choose_s') & source_core.group.eq('S')].copy()
    if len(s) != 1:
        raise ValueError('Expected one overall p_choose_s/S summary')
    s['metric'] = 'choice_probability'
    e = s.copy()
    e['group'] = 'E'
    for col in ('empirical', 'ppc_mean', 'ppc_median'):
        e[col] = 1.0 - s[col].to_numpy(dtype=float)
    e['ppc_pi95_low'] = 1.0 - s['ppc_pi95_high'].to_numpy(dtype=float)
    e['ppc_pi95_high'] = 1.0 - s['ppc_pi95_low'].to_numpy(dtype=float)
    e['ppc_minus_empirical'] = e.ppc_mean - e.empirical
    e['absolute_error'] = e.ppc_minus_empirical.abs()
    e['empirical_in_ppc_95'] = e.empirical.between(e.ppc_pi95_low, e.ppc_pi95_high)
    return pd.concat((e, s), ignore_index=True)


def model_summary(sources: dict, median_cache: pd.DataFrame, study: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cores, curves = [], []
    for model, source in sources.items():
        median = median_summary(median_cache, model)
        choice = _choice_probability_rows(source['core'])
        core = pd.concat((source['core'], median, choice), ignore_index=True)
        quint = source['curves'].copy()
        for frame in (core, quint):
            frame.insert(0, 'study', study)
            frame.insert(1, 'model', model)
            frame.insert(2, 'model_label', LABELS[model])
            frame['squared_error'] = frame.ppc_minus_empirical ** 2
        cores.append(core)
        curves.append(quint)
    return pd.concat(cores, ignore_index=True), pd.concat(curves, ignore_index=True)


def _select(frame: pd.DataFrame, metric: str, group: str | None, positions: tuple) -> pd.DataFrame:
    part = frame.loc[frame.metric.eq(metric)].copy()
    if group is not None:
        part = part.loc[part.group.eq(group)]
    if isinstance(positions[0], int):
        part = part.set_index('quintile').reindex(positions)
    else:
        part = part.set_index('group').reindex(positions)
    needed = ['empirical', 'ppc_mean', 'ppc_pi95_low', 'ppc_pi95_high']
    if len(part) != len(positions) or part[needed].isna().any().any():
        raise ValueError(f'Incomplete PPC summary: {metric}/{group}')
    return part


def build_rankings(core: pd.DataFrame, curves: pd.DataFrame) -> pd.DataFrame:
    entries = []
    for metric, group, _, _, _, positions, _ in METRICS:
        frame = curves if isinstance(positions[0], int) else core
        for model in MODELS:
            part = _select(frame.loc[frame.model.eq(model)], metric, group, positions)
            error = part.ppc_mean.to_numpy(float) - part.empirical.to_numpy(float)
            entries.append(dict(study=part.study.iloc[0], model=model, model_label=LABELS[model],
                                comparison=metric, n_points=len(part),
                                rmse=float(np.sqrt(np.mean(error ** 2))),
                                mae=float(np.mean(np.abs(error))),
                                covered_points_95=int(np.sum(part.empirical.between(
                                    part.ppc_pi95_low, part.ppc_pi95_high)))))
    ranks = pd.DataFrame(entries)
    ranks['rank_by_rmse'] = ranks.groupby('comparison')['rmse'].rank(method='min', ascending=True).astype(int)
    return ranks.sort_values(['comparison', 'rank_by_rmse', 'model']).reset_index(drop=True)


def save_plot(fig, stem: Path):
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix('.png'), dpi=300, facecolor='white', bbox_inches='tight')
    fig.savefig(stem.with_suffix('.pdf'), facecolor='white', bbox_inches='tight')
    plt.close(fig)




def _small_figure_legend(ax):
    """Place the multi-model legend below the panel, avoiding data occlusion."""
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.29),
              frameon=False, fontsize=FONT_LEGEND, ncol=2,
              columnspacing=0.9, handlelength=1.1, handletextpad=0.45)


def _plot_generic(core: pd.DataFrame, curves: pd.DataFrame, study: str,
                  spec: tuple, output: Path, show_bands: bool = True):
    metric, group, title, xlabel, ylabel, positions, _ = spec
    frame = curves if isinstance(positions[0], int) else core
    x = np.arange(len(positions), dtype=float)
    is_probability = metric.startswith('p_choose_s') or metric in ('choice_probability', 'accuracy')
    scale = 100.0 if metric in ('choice_probability', 'accuracy') else 1.0
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    empirical = None
    y_limits = []

    for model in MODELS:
        rows = _select(frame.loc[frame.model.eq(model)], metric, group, positions)
        observed = rows.empirical.to_numpy(dtype=float) * scale
        predicted = rows.ppc_mean.to_numpy(dtype=float) * scale
        low = rows.ppc_pi95_low.to_numpy(dtype=float) * scale
        high = rows.ppc_pi95_high.to_numpy(dtype=float) * scale
        if empirical is None:
            empirical = observed
        elif not np.allclose(empirical, observed, atol=1e-6, rtol=0):
            raise ValueError(f'Observed values differ between models: {metric}')
        color = COLORS[model]
        ax.plot(x, predicted, color=color, linewidth=2.0, markersize=5.5,
                marker='o', label=LABELS[model], zorder=3)
        if show_bands:
            ax.fill_between(x, low, high, alpha=.08, color=color, linewidth=0, zorder=1)
            y_limits.extend((low, high))
        y_limits.append(predicted)

    ax.plot(x, empirical, color=OBS_COLOR, linestyle='--', marker='s',
            linewidth=2.0, markersize=5.5, label='Observed', zorder=5)
    y_limits.append(empirical)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in positions], fontsize=FONT_TICKS)
    ax.set_xlabel(xlabel, fontsize=FONT_AXIS)
    ax.set_ylabel(ylabel, fontsize=FONT_AXIS)
    ax.set_title(title, fontsize=FONT_TITLE, pad=9)
    _style_axes(ax)
    _set_limits(ax, y_limits, percent=(metric in ('choice_probability', 'accuracy')))
    _small_figure_legend(ax)
    fig.subplots_adjust(left=0.20, right=0.98, top=0.89, bottom=0.33)
    save_plot(fig, output)


def _plot_rt_quintile_compare(curves: pd.DataFrame, study: str, output: Path, show_bands: bool = True):
    x = np.arange(1, 6, dtype=float)
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    first = _select(curves.loc[curves.model.eq(MODELS[0])], 'p_choose_s_by_rt_quintile', 'S', (1, 2, 3, 4, 5))
    empirical_s = first.empirical.to_numpy(float) * 100.0
    ax.bar(x, empirical_s, width=0.64, color=EMP_S, edgecolor=EDGE, linewidth=1.2,
           alpha=.92, label='Empirical S', zorder=1)
    if 'empirical_sem' in first:
        ax.errorbar(x, empirical_s, yerr=first.empirical_sem.to_numpy(float) * 100,
                    fmt='none', ecolor=EDGE, capsize=3, elinewidth=1.3, zorder=3)
    ax.axhline(50, color='#111111', linestyle=(0, (1.2, 4.0)), linewidth=2.0, zorder=0)

    limits = [empirical_s]
    for model in MODELS:
        rows = _select(curves.loc[curves.model.eq(model)], 'p_choose_s_by_rt_quintile', 'S', (1, 2, 3, 4, 5))
        pred = rows.ppc_mean.to_numpy(float) * 100.0
        low = rows.ppc_pi95_low.to_numpy(float) * 100.0
        high = rows.ppc_pi95_high.to_numpy(float) * 100.0
        color = COLORS[model]
        ax.plot(x, pred, color=color, marker='o', linewidth=1.9, markersize=5.3,
                label=LABELS[model], zorder=4)
        ax.errorbar(x, pred, yerr=[pred - low, high - pred], fmt='none',
                    ecolor=color, elinewidth=1.3, capsize=3, zorder=4)
        if show_bands:
            ax.fill_between(x, low, high, alpha=.06, color=color, linewidth=0, zorder=2)
        limits.extend([pred, low, high])

    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in range(1, 6)], fontsize=FONT_TICKS)
    ax.set_xlabel('RT quintile (fast to slow)', fontsize=FONT_AXIS)
    ax.set_ylabel('Probability (%)', fontsize=FONT_AXIS)
    ax.set_title('Choice by RT quintile', fontsize=FONT_TITLE, pad=9)
    _style_axes(ax)
    _set_limits(ax, limits, percent=True, include_half=True)
    _small_figure_legend(ax)
    fig.subplots_adjust(left=0.20, right=0.98, top=0.89, bottom=0.33)
    save_plot(fig, output)


def _plot_dwell_compare(curves: pd.DataFrame, study: str, output: Path, show_bands: bool = True):
    x = np.arange(1, 6, dtype=float)
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    first = _select(curves.loc[curves.model.eq(MODELS[0])], 'p_choose_s_by_dwell_quintile', 'S', (1, 2, 3, 4, 5))
    empirical = first.empirical.to_numpy(float)
    ax.axhline(0.5, color='#7a7a7a', linestyle=(0, (1.2, 3.2)), linewidth=1.5, zorder=0)
    ax.plot(x, empirical, color=EMP_S, marker='o', linewidth=2.3, markersize=6,
            label='Empirical', zorder=3)
    if {'empirical_ci_low', 'empirical_ci_high'}.issubset(first.columns):
        ell = first.empirical_ci_low.to_numpy(float)
        ehi = first.empirical_ci_high.to_numpy(float)
        if np.isfinite(ell).all() and np.isfinite(ehi).all():
            ax.errorbar(x, empirical, yerr=[empirical - ell, ehi - empirical],
                        fmt='none', ecolor=EMP_S, elinewidth=1.35, capsize=3, zorder=3)

    limits = [empirical]
    for model in MODELS:
        rows = _select(curves.loc[curves.model.eq(model)], 'p_choose_s_by_dwell_quintile', 'S', (1, 2, 3, 4, 5))
        pred = rows.ppc_mean.to_numpy(float)
        low = rows.ppc_pi95_low.to_numpy(float)
        high = rows.ppc_pi95_high.to_numpy(float)
        color = COLORS[model]
        ax.plot(x, pred, color=color, marker='o', linewidth=1.9, markersize=5.3,
                label=LABELS[model], zorder=4)
        ax.errorbar(x, pred, yerr=[pred - low, high - pred], fmt='none',
                    ecolor=color, elinewidth=1.3, capsize=3, zorder=4)
        if show_bands:
            ax.fill_between(x, low, high, alpha=.06, color=color, linewidth=0, zorder=2)
        limits.extend([pred, low, high])

    ax.set_xticks(x)
    ax.set_xticklabels(DWELL_LABELS, fontsize=FONT_TICKS)
    ax.set_xlabel('Dwell advantage (S − E)', fontsize=FONT_AXIS)
    ax.set_ylabel('P(Choose Symbolic)', fontsize=FONT_AXIS)
    ax.set_title('Choice by dwell-time advantage', fontsize=FONT_TITLE, pad=9)
    _style_axes(ax)
    _set_limits(ax, limits, is_probability=True, include_half=True)
    _small_figure_legend(ax)
    fig.subplots_adjust(left=0.20, right=0.98, top=0.89, bottom=0.33)
    save_plot(fig, output)


def plot_comparison(core: pd.DataFrame, curves: pd.DataFrame, study: str,
                    spec: tuple, output: Path, show_bands: bool = True):
    metric = spec[0]
    if metric == 'p_choose_s_by_rt_quintile':
        _plot_rt_quintile_compare(curves, study, output, show_bands)
    elif metric == 'p_choose_s_by_dwell_quintile':
        _plot_dwell_compare(curves, study, output, show_bands)
    else:
        _plot_generic(core, curves, study, spec, output, show_bands)


def run_comparison(study: str, root: Path = ROOT, *, show_bands: bool = True,
                   rebuild_median_cache: bool = False) -> Path:
    if study not in ('study1', 'study2'):
        raise ValueError('study must be study1 or study2')
    sources, audit = load_sources(study, root)
    outdir = root / 'studies' / study / 'results' / 'ppc_compare' / 'four_model'
    outdir.mkdir(parents=True, exist_ok=True)
    medians = load_or_calculate_medians(study, sources, outdir, rebuild=rebuild_median_cache)
    core, curves = model_summary(sources, medians, study)
    rankings = build_rankings(core, curves)
    prefix = f'{study}_four_model'
    _atomic_csv(audit, outdir / 'data' / f'{prefix}_input_audit.csv')
    _atomic_csv(core, outdir / f'{prefix}_comparison_summary.csv')
    _atomic_csv(curves, outdir / f'{prefix}_comparison_quintiles.csv')
    _atomic_csv(rankings, outdir / f'{prefix}_model_rankings.csv')
    for metric in METRICS:
        plot_comparison(core, curves, study, metric,
                        outdir / 'plots' / f'{prefix}_{metric[6]}', show_bands)
    details = dict(
        study=study,
        models=list(MODELS),
        ppc_draws_per_model=int(audit.ppc_draws.iloc[0]),
        generated_figures=[v[6] for v in METRICS],
        median_rt_definition='participant mean of participant-level median RT; E/S medians conditional on choice',
        model_colours=COLORS,
        empirical_colour=OBS_COLOR,
        posterior_predictive_intervals='2.5–97.5% across replicated datasets',
        comparison_type='in-sample PPC diagnostics, not out-of-sample model selection',
        choice_probability_definition='P(E)=1-P(S), calculated from aligned PPC draw summaries',
        quintile_method='posthoc R-style participant RT ntile(5); unique-value DTA quantile cut(5)',
        empirical_quintile_intervals='RT: participant SEM; dwell: participant 95% t confidence interval',
        plotted_proportions_as_percent=True,
        created_utc=datetime.now(timezone.utc).isoformat(),
    )
    (outdir / f'{prefix}_comparison_run.json').write_text(json.dumps(details, indent=2) + '\n', encoding='utf-8')
    print(f'PASS FOUR-MODEL PPC COMPARISON {study}: {len(METRICS)} figures, 3 comparison tables, no posterior predictive simulations run', flush=True)
    return outdir


def main(argv=None):
    parser = argparse.ArgumentParser(description='Compare four identity-coded ES model PPCs from already-saved data')
    parser.add_argument('--study', required=True, choices=['study1', 'study2'])
    parser.add_argument('--no-bands', action='store_true', help='Hide 95% predictive ribbons (unchanged tables)')
    parser.add_argument('--rebuild-median-cache', action='store_true', help='Recalculate median RT from saved trial archives')
    args = parser.parse_args(argv)
    run_comparison(args.study, show_bands=not args.no_bands,
                   rebuild_median_cache=args.rebuild_median_cache)


if __name__ == '__main__':
    main()
