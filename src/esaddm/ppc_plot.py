"""Offline PPC tables and talk-ready plots from saved generated PPC summaries.

Never loads HDDM, never simulates PPC, never changes generation outputs.
Rerun at any time to change colours, typography, group summaries and figures.
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
from scipy.stats import t as student_t
import numpy as np
import pandas as pd

from .models import ROOT, load_models
from .ppc import (
    OBS_E, OBS_S, PPC_E, PPC_S, OBS_OTHER, PPC_OTHER,
    QUINTILES, IDENTITY_MODELS, model_prefix, summarise_metrics, subject_boxplot_data,
)

# Slightly larger typography throughout.
FONT_TITLE = 18
FONT_PANEL = 18
FONT_AXIS = 18
FONT_TICKS = 18
FONT_LEGEND = 16
FIG_WIDTH = 4.8
FIG_HEIGHT = 4.2

# Stronger colours for empirical bars/lines.
EMP_E = '#19B5E5'
EMP_S = '#A13AD8'
PRED_E = PPC_E
PRED_S = PPC_S
EDGE = '#000000'
MIDLINE = '#111111'

RT_LABELS = ['1', '2', '3', '4', '5']
DWELL_LABELS = ['E>>S', 'E>S', 'E~S', 'S>E', 'S>>E']


def style_plot(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.6)
    ax.spines['bottom'].set_linewidth(1.6)
    ax.tick_params(labelsize=FONT_TICKS, width=1.3)
    ax.grid(False)


def save_plot(fig, output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix('.png'), dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(fig)


def _percent_formatter(x, pos):
    return f'{int(round(x))}%'


def _probability_limits(ax, values, scale='proportion', include_half=False):
    arr = np.concatenate([np.asarray(v, dtype=float).ravel() for v in values])
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return
    if scale == 'percent':
        lo = max(0.0, np.floor((arr.min() - 3.0) / 5.0) * 5.0)
        hi = min(100.0, np.ceil((arr.max() + 3.0) / 5.0) * 5.0)
        if include_half:
            lo = min(lo, 45.0)
            hi = max(hi, 55.0)
        if hi - lo < 15:
            mid = (hi + lo) / 2.0
            lo = max(0.0, mid - 10.0)
            hi = min(100.0, mid + 10.0)
        ax.set_ylim(lo, hi)
        ax.yaxis.set_major_formatter(FuncFormatter(_percent_formatter))
    else:
        lo = max(0.0, arr.min() - 0.04)
        hi = min(1.0, arr.max() + 0.04)
        if include_half:
            lo = min(lo, 0.45)
            hi = max(hi, 0.55)
        if hi - lo < 0.20:
            mid = (hi + lo) / 2.0
            lo = max(0.0, mid - 0.12)
            hi = min(1.0, mid + 0.12)
        ax.set_ylim(lo, hi)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))


def _curve_slice(curves, metric):
    part = curves.loc[curves.metric.eq(metric)].sort_values('quintile')
    if len(part) != 5 or not np.array_equal(part.quintile.to_numpy(), np.array(QUINTILES)):
        raise ValueError(f'Incomplete quintile PPC curve for {metric}')
    return part


def _empirical_error_columns(rows: pd.DataFrame):
    # Use empirical uncertainty if it already exists in the saved summary.
    candidates = [
        ('empirical_ci_low', 'empirical_ci_high'),
        ('empirical_se', 'se'),
        ('empirical_sem', 'sem'),
        ('empirical_sd', 'sd'),
        ('empirical_low', 'empirical_high'),
        ('empirical_ci_low', 'empirical_ci_high'),
        ('empirical_pi95_low', 'empirical_pi95_high'),
    ]
    cols = set(rows.columns)
    for low_name, high_name in candidates:
        if low_name in cols and high_name in cols:
            low = rows[low_name].to_numpy(dtype=float)
            high = rows[high_name].to_numpy(dtype=float)
            if high_name in ('se', 'sem', 'sd'):
                continue
            return low, high
    if 'empirical_se' in cols:
        se = rows['empirical_se'].to_numpy(dtype=float)
        center = rows['empirical'].to_numpy(dtype=float)
        return center - se, center + se
    if 'empirical_sem' in cols:
        se = rows['empirical_sem'].to_numpy(dtype=float)
        center = rows['empirical'].to_numpy(dtype=float)
        return center - se, center + se
    return None, None


def plot_boxplots(subject_data: pd.DataFrame, study: str, model: str, output: Path,
                  metric: str):
    data = subject_data.loc[subject_data.metric.eq(metric)].copy()
    groups = ['Overall'] if model == 'accuracy_ee' else ['E', 'S']
    if model != 'accuracy_ee' and metric == 'accuracy':
        groups = ['E', 'S']

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    for i, group in enumerate(groups):
        sub = data.loc[data.group.eq(group)].sort_values('subj_idx')
        if sub.empty:
            raise ValueError(f'Missing {metric} group {group}')
        emp_color = {'E': EMP_E, 'S': EMP_S}.get(group, OBS_OTHER)
        pred_color = {'E': PRED_E, 'S': PRED_S}.get(group, PPC_OTHER)
        for offset, name, color in [(-.18, 'empirical_value', emp_color),
                                    (.18, 'ppc_subject_mean', pred_color)]:
            x = i + offset
            bp = ax.boxplot(
                sub[name].to_numpy(), positions=[x], widths=.28,
                patch_artist=True, showfliers=False,
                medianprops={'color': '#252525', 'linewidth': 2.5},
                whiskerprops={'color': '#444444', 'linewidth': 1.3},
                capprops={'color': '#444444', 'linewidth': 1.3},
            )
            bp['boxes'][0].set_facecolor(color)
            bp['boxes'][0].set_alpha(.88)
            bp['boxes'][0].set_edgecolor('#1f1f1f')
            bp['boxes'][0].set_linewidth(1.2)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, fontsize=FONT_AXIS)
    ax.set_ylabel('Mean RT (seconds)' if metric == 'mean_rt' else 'Mean accuracy (0–1)',
                  fontsize=FONT_AXIS)
    if metric == 'accuracy':
        ax.set_ylim(0, 1.05)
    ax.set_title('Mean RT' if metric == 'mean_rt' else 'Mean accuracy',
                 fontsize=FONT_TITLE, pad=8)

    if model == 'accuracy_ee':
        legend_items = [
            mpatches.Patch(facecolor=OBS_OTHER, edgecolor='#1f1f1f', label='Empirical'),
            mpatches.Patch(facecolor=PPC_OTHER, edgecolor='#1f1f1f', label='PPC'),
        ]
    else:
        legend_items = [
            mpatches.Patch(facecolor=EMP_E, edgecolor='#1f1f1f', label='Empirical E'),
            mpatches.Patch(facecolor=PRED_E, edgecolor='#1f1f1f', label='PPC E'),
            mpatches.Patch(facecolor=EMP_S, edgecolor='#1f1f1f', label='Empirical S'),
            mpatches.Patch(facecolor=PRED_S, edgecolor='#1f1f1f', label='PPC S'),
        ]
    ax.legend(handles=legend_items, fontsize=FONT_LEGEND, frameon=False,
              ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.23),
              columnspacing=0.7, handletextpad=0.3)
    style_plot(ax)
    fig.subplots_adjust(left=0.20, right=0.98, top=0.90, bottom=0.32)
    save_plot(fig, output)


def plot_choice_rt(curves: pd.DataFrame, study: str, model: str, output: Path):
    rows = _curve_slice(curves, 'p_choose_s_by_rt_quintile')
    x = np.arange(1, 6, dtype=float)
    width = 0.36

    empirical_s = rows.empirical.to_numpy(float) * 100.0
    empirical_e = 100.0 - empirical_s
    ppc_s = rows.ppc_mean.to_numpy(float) * 100.0
    ppc_e = 100.0 - ppc_s
    low_s = rows.ppc_pi95_low.to_numpy(float) * 100.0
    high_s = rows.ppc_pi95_high.to_numpy(float) * 100.0
    low_e = 100.0 - high_s
    high_e = 100.0 - low_s

    fig, ax = plt.subplots(figsize=(FIG_WIDTH + 0.3, FIG_HEIGHT + 0.3))
    ax.bar(x - width / 2.0, empirical_e, width=width, color=EMP_E,
           edgecolor=EDGE, linewidth=1.2, alpha=.96, zorder=1)
    ax.bar(x + width / 2.0, empirical_s, width=width, color=EMP_S,
           edgecolor=EDGE, linewidth=1.2, alpha=.96, zorder=1)
    if 'empirical_sem' in rows:
        sem = rows.empirical_sem.to_numpy(float) * 100.0
        ax.errorbar(x - width / 2.0, empirical_e, yerr=sem, fmt='none',
                    ecolor=EDGE, capsize=3, elinewidth=1.3, zorder=3)
        ax.errorbar(x + width / 2.0, empirical_s, yerr=sem, fmt='none',
                    ecolor=EDGE, capsize=3, elinewidth=1.3, zorder=3)

    ax.plot(x - width / 2.0, ppc_e, color=PRED_E, marker='o', linewidth=3.0,
            markersize=6.0, label='PPC E', zorder=4)
    ax.errorbar(x - width / 2.0, ppc_e, yerr=[ppc_e - low_e, high_e - ppc_e],
                fmt='none', ecolor=PRED_E, elinewidth=2.0, capsize=5, zorder=4)
    ax.plot(x + width / 2.0, ppc_s, color=PRED_S, marker='o', linewidth=3.0,
            markersize=6.0, label='PPC S', zorder=4)
    ax.errorbar(x + width / 2.0, ppc_s, yerr=[ppc_s - low_s, high_s - ppc_s],
                fmt='none', ecolor=PRED_S, elinewidth=2.0, capsize=5, zorder=4)

    ax.axhline(50, color=MIDLINE, linestyle=(0, (1.2, 4.0)), linewidth=2.0, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in range(1, 6)], fontsize=FONT_TICKS)
    ax.set_xlabel('RT quintile (fast to slow)', fontsize=FONT_AXIS)
    ax.set_ylabel('Probability (%)', fontsize=FONT_AXIS)
    ax.set_title('Choice by RT quintile', fontsize=FONT_TITLE, pad=8)
    _probability_limits(ax, [empirical_e, empirical_s, low_e, high_e, low_s, high_s],
                        scale='percent', include_half=True)

    legend_handles = [
        mpatches.Patch(facecolor=EMP_E, edgecolor=EDGE, label='Empirical E'),
        mpatches.Patch(facecolor=EMP_S, edgecolor=EDGE, label='Empirical S'),
        Line2D([0], [0], color=PRED_E, marker='o', linewidth=2.0, markersize=8.0, label='PPC E'),
        Line2D([0], [0], color=PRED_S, marker='o', linewidth=2.0, markersize=8.0, label='PPC S'),
    ]
    ax.legend(handles=legend_handles, fontsize=FONT_LEGEND, frameon=False,
              ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.30),
              columnspacing=0.8, handletextpad=0.4)
    style_plot(ax)
    fig.subplots_adjust(left=0.19, right=0.98, top=0.90, bottom=0.34)
    save_plot(fig, output)




def plot_dwell(curves: pd.DataFrame, study: str, model: str, output: Path):
    rows = _curve_slice(curves, 'p_choose_s_by_dwell_quintile')
    x = np.arange(1, 6, dtype=float)
    empirical = rows.empirical.to_numpy(float)
    pred = rows.ppc_mean.to_numpy(float)
    pred_low = rows.ppc_pi95_low.to_numpy(float)
    pred_high = rows.ppc_pi95_high.to_numpy(float)
    emp_low, emp_high = _empirical_error_columns(rows)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    ax.axhline(0.5, color='#7a7a7a', linestyle=(0, (1.2, 3.2)), linewidth=1.6, zorder=0)

    # empirical line first
    ax.plot(x, empirical, color=EMP_S, marker='o', linewidth=3.2,
            markersize=9.0, label='Empirical', zorder=3)
    if emp_low is not None and emp_high is not None:
        ax.errorbar(x, empirical, yerr=[empirical - emp_low, emp_high - empirical],
                    fmt='none', ecolor=EMP_S, elinewidth=1.9, capsize=4, zorder=3)

    ax.plot(x, pred, color=PRED_S, marker='o', linewidth=3.0,
            markersize=6.0, label='PPC', zorder=4)
    ax.errorbar(x, pred, yerr=[pred - pred_low, pred_high - pred],
                fmt='none', ecolor=PRED_S, elinewidth=2.0, capsize=5, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(DWELL_LABELS, fontsize=FONT_TICKS)
    ax.set_xlabel('Dwell advantage (S − E)', fontsize=FONT_AXIS)
    ax.set_ylabel('P(Choose Symbolic)', fontsize=FONT_AXIS)
    ax.set_title('Choice by dwell-time advantage', fontsize=FONT_TITLE, pad=8)
    _probability_limits(ax, [empirical, pred, pred_low, pred_high],
                        scale='proportion', include_half=True)
    ax.legend(frameon=False, fontsize=FONT_LEGEND, loc='best')
    style_plot(ax)
    fig.subplots_adjust(left=0.20, right=0.98, top=0.9, bottom=0.24)
    save_plot(fig, output)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f'Missing saved PPC data: {path}')
    data = pd.read_csv(path)
    if data.empty:
        raise ValueError(f'Empty saved PPC data: {path}')
    return data


def _check_ppc_saved(study: str, model: str, result_dir: Path):
    prefix = model_prefix(model, study)
    path = result_dir / f'{prefix}_ppc_run.json'
    if not path.is_file():
        raise FileNotFoundError(f'Missing PPC generation manifest: {path}')
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if (manifest.get('status') != 'completed' or
            manifest.get('study') != study or manifest.get('model') != model):
        raise ValueError('PPC generation is not completed for the requested study/model')
    generated = manifest.get('ppc_draws_generated')
    if not isinstance(generated, int) or generated <= 0:
        raise ValueError('No valid number of generated PPC datasets in manifest')
    return manifest


# R-aligned re-aggregation of existing saved PPC trials (NO model loading/simulation).
# These exclusions match the identity-coded aDDM fitting samples.
FIT_EXCLUDED = {
    'study1': frozenset({1, 4, 5, 6, 14, 99}),
    'study2': frozenset({2, 6, 9, 14, 18, 20, 26}),
}
RSTYLE_VERSION = 3
RT_RANGE = (.250, 10.0)  # Study 1 behavioural R script; also applied to Study 2
RSTYLE_METRICS = ('p_choose_s_by_rt_quintile', 'p_choose_s_by_dwell_quintile')
TRIAL_KEYS = ['sub_id', 'phase', 'trial']


def _ntile_r(x: pd.Series) -> pd.Series:
    """Replicate dplyr::ntile(x, 5): larger groups first, break ties by row order."""
    v = pd.to_numeric(x, errors='coerce')
    out = pd.Series(pd.NA, index=x.index, dtype='Int64')
    ordered = v.dropna().sort_values(kind='stable').index
    n = len(ordered)
    if not n:
        return out
    big, small = divmod(n, 5)
    ranks = np.arange(n)
    # The first `small` groups have size `big+1`; remaining groups size `big`.
    first_count = small * (big + 1)
    if big:
        bins = np.where(ranks < first_count, ranks // (big + 1) + 1,
                        small + (ranks - first_count) // big + 1)
    else:
        bins = ranks + 1
    out.loc[ordered] = bins.astype(int)
    return out


def _dwell_r(x: pd.Series) -> pd.Series:
    """Replicate R: quantile(sort(unique(finite_dta)),0:.2:1), then cut()."""
    v = pd.to_numeric(x, errors='coerce')
    out = pd.Series(pd.NA, index=x.index, dtype='Int64')
    unique = np.sort(v.loc[np.isfinite(v)].unique())
    if len(unique) < 5:
        return out   # exactly the R make_dta_bins() fallback
    edges = np.quantile(unique, np.linspace(0, 1, 6), method='linear')
    if not np.all(np.diff(edges) > 0):
        raise ValueError('Nonunique participant DTA cutpoints; R cut() also fails')
    out = pd.cut(v, bins=edges, labels=list(QUINTILES),
                 right=True, include_lowest=True).astype('Int64')
    return out


def _empirical_bin_summary(obs: pd.DataFrame, metric: str) -> pd.DataFrame:
    sub = (obs.groupby(['subj_idx', 'quintile'], observed=True)
           .response.mean().reset_index(name='p_s'))
    agg = (sub.groupby('quintile', observed=True).p_s
           .agg(['mean', 'std', 'count']).reindex(QUINTILES))
    if agg['count'].isna().any() or agg['count'].lt(2).any():
        raise ValueError(f'Insufficient empirical participants per {metric} quintile')
    n = agg['count'].to_numpy(dtype=int)
    sem = agg['std'].to_numpy(float) / np.sqrt(n)
    margin = student_t.ppf(.975, n-1) * sem
    mean = agg['mean'].to_numpy(float)
    return pd.DataFrame({'metric': metric, 'group': 'S', 'quintile': QUINTILES,
                         'empirical': mean, 'empirical_sem': sem,
                         'empirical_ci_low': mean-margin, 'empirical_ci_high': mean+margin,
                         'empirical_n_subjects': n})


def _rebin_signature(study: str, model: str, directory: Path, manifest: dict) -> dict:
    prefix = model_prefix(model, study)
    trialdir = directory / 'data' / 'trials'
    archives = [trialdir / f'{prefix}_observed_trials.csv.gz'] + [
        trialdir / f'{prefix}_simulated_chain{i}.csv.gz'
        for i in range(1, int(manifest.get('chains', 5)) + 1)]
    fingerprint = []
    for p in archives:
        stat = p.stat()  # Require every source archive to exist.
        fingerprint.append([p.name, stat.st_size, stat.st_mtime_ns])
    return {'version': RSTYLE_VERSION, 'study': study, 'model': model,
            'fit_exclusions': sorted(FIT_EXCLUDED[study]), 'rt_range_s': list(RT_RANGE),
            'expected_draws': manifest['ppc_draws_generated'], 'files': fingerprint}


def rstyle_curves_from_archives(study: str, model: str, result_dir: Path,
                                manifest: dict | None = None) -> pd.DataFrame:
    """Recalculate R-style curves from existing trial archives, cache, and reuse.

    Both ppc_plot.py and ppc_compare.py call this exact function. No HDDM run.
    The original generated PPC metrics and compressed archives are never modified.
    """
    if study not in FIT_EXCLUDED:
        raise ValueError(f'Unknown study {study}')
    manifest = manifest or _check_ppc_saved(study, model, result_dir)
    prefix = model_prefix(model, study)
    signature = _rebin_signature(study, model, result_dir, manifest)
    cache_dir = result_dir / 'data' / 'rstyle_posthoc'
    cache_csv = cache_dir / f'{prefix}_rstyle_curves.csv'
    cache_manifest = cache_dir / f'{prefix}_rstyle_cache.json'
    if cache_csv.is_file() and cache_manifest.is_file():
        if json.loads(cache_manifest.read_text()) == signature:
            print(f'REUSING R-STYLE PPC CURVES: {study}/{model}', flush=True)
            return pd.read_csv(cache_csv)

    trialdir = result_dir / 'data' / 'trials'
    observed = pd.read_csv(trialdir / f'{prefix}_observed_trials.csv.gz')
    required = set(TRIAL_KEYS + ['subj_idx', 'response', 'rt_value',
                                 'dwell_ms_s_minus_e', 'Dwell_E', 'Dwell_S'])
    if not required.issubset(observed):
        raise ValueError(f'Missing observed trial fields {sorted(required-set(observed))}')
    if observed.duplicated(TRIAL_KEYS).any() or not observed.phase.astype(str).eq('ES').all():
        raise ValueError('Observed ES trial identity missing, duplicated, or wrong phase')
    observed['sub_id_int'] = pd.to_numeric(observed.sub_id, errors='raise')
    present = sorted(set(observed.sub_id_int.dropna().astype(int)) & FIT_EXCLUDED[study])
    if present:
        raise ValueError(f'{study}/{model}: these excluded participants appear in fitted PPC '
                         f'archives: {present}. Refusing to mix unfitted observations.')
    observed['response'] = pd.to_numeric(observed.response, errors='raise')
    if not observed.response.isin([0, 1]).all():
        raise ValueError('Expected identity-coded ES response E=0, S=1')
    observed['rt_value'] = pd.to_numeric(observed.rt_value, errors='raise').abs()
    observed['dwell_ms_s_minus_e'] = pd.to_numeric(observed.dwell_ms_s_minus_e, errors='coerce')
    observed['Dwell_E'] = pd.to_numeric(observed.Dwell_E, errors='coerce')
    observed['Dwell_S'] = pd.to_numeric(observed.Dwell_S, errors='coerce')
    rt = observed.loc[observed.rt_value.between(*RT_RANGE)].copy()
    rt['quintile'] = rt.groupby('subj_idx', group_keys=False).rt_value.transform(_ntile_r)
    dw = observed.loc[np.isfinite(observed.dwell_ms_s_minus_e)
                      & observed.Dwell_E.gt(-1) & observed.Dwell_S.gt(-1)].copy()
    dw['quintile'] = dw.groupby('subj_idx', group_keys=False).dwell_ms_s_minus_e.transform(_dwell_r)
    missing_dw = int(dw.quintile.isna().sum())
    dw = dw.loc[dw.quintile.notna()].copy()
    if rt.empty or dw.empty:
        raise ValueError('No valid RT or dwell trials after R filters')
    empirical = pd.concat([_empirical_bin_summary(rt, RSTYLE_METRICS[0]),
                           _empirical_bin_summary(dw, RSTYLE_METRICS[1])], ignore_index=True)
    mapping = observed[TRIAL_KEYS + ['subj_idx']].copy()
    mapping['dwell_quintile_r'] = pd.Series(pd.NA, index=observed.index, dtype='Int64')
    mapping.loc[dw.index, 'dwell_quintile_r'] = dw.quintile.to_numpy()
    metrics = []
    draw_names = set()
    for ci in range(1, int(manifest.get('chains', 5)) + 1):
        path = trialdir / f'{prefix}_simulated_chain{ci}.csv.gz'
        print(f'REBIN SAVED PPC: {study}/{model} chain {ci}...', flush=True)
        simulated = pd.read_csv(path, usecols=['ppc_draw', *TRIAL_KEYS, 'subj_idx',
                                                'rt_value', 'response_sampled'])
        simulated = simulated.merge(mapping, on=TRIAL_KEYS, how='left',
                                    validate='many_to_one', suffixes=('', '_observed'))
        if simulated.subj_idx_observed.isna().any() or not simulated.subj_idx.eq(simulated.subj_idx_observed).all():
            raise ValueError(f'Simulated trials misaligned with observed trial identities: {path}')
        simulated['rt_value'] = pd.to_numeric(simulated.rt_value, errors='raise').abs()
        simulated['response_sampled'] = pd.to_numeric(simulated.response_sampled, errors='raise')
        if not simulated.response_sampled.isin([0, 1]).all():
            raise ValueError('PPC sampled response must be 0/1')
        for draw, frame in simulated.groupby('ppc_draw', sort=False):
            if draw in draw_names or len(frame) != len(observed) or frame.duplicated(TRIAL_KEYS).any():
                raise ValueError(f'Duplicate or incomplete PPC trial dataset: {draw}')
            draw_names.add(draw)
            rt_draw = frame.loc[frame.rt_value.between(*RT_RANGE)].copy()
            rt_draw['quintile'] = (rt_draw.groupby('subj_idx', group_keys=False).rt_value
                                  .transform(_ntile_r))
            dw_draw = frame.loc[frame.dwell_quintile_r.notna()].copy()
            dw_draw['quintile'] = dw_draw.dwell_quintile_r.astype('int64')
            for metric, part in zip(RSTYLE_METRICS, (rt_draw, dw_draw)):
                subject = part.groupby(['subj_idx', 'quintile'], observed=True).response_sampled.mean()
                group = subject.groupby('quintile').mean().reindex(QUINTILES)
                if group.isna().any():
                    raise ValueError(f'No participants contributing to some quintiles: {draw} {metric}')
                metrics.extend({'draw': str(draw), 'metric': metric, 'quintile': q,
                                'value': float(value)} for q, value in group.items())
    predictions = pd.DataFrame(metrics)
    if len(draw_names) != int(manifest['ppc_draws_generated']):
        raise ValueError('Saved PPC draw count does not match the manifest')
    grouped = predictions.groupby(['metric', 'quintile']).value
    pred = grouped.agg(ppc_mean='mean', ppc_sd='std', ppc_median='median', n_ppc='size').reset_index()
    pi = grouped.quantile([.025, .975]).unstack().rename(
        columns={.025: 'ppc_pi95_low', .975: 'ppc_pi95_high'}).reset_index()
    output = empirical.merge(pred.merge(pi, on=['metric', 'quintile']),
                             on=['metric', 'quintile'], validate='one_to_one')
    if output.isna().any().any():
        raise ValueError('Missing R-style PPC summary values')
    output['ppc_minus_empirical'] = output.ppc_mean - output.empirical
    output['absolute_error'] = output.ppc_minus_empirical.abs()
    output['empirical_in_ppc_95'] = output.empirical.between(
        output.ppc_pi95_low, output.ppc_pi95_high)
    output = output.sort_values(['metric', 'quintile']).reset_index(drop=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache_csv.with_suffix('.csv.part')
    output.to_csv(tmp, index=False, float_format='%.12g')
    tmp.replace(cache_csv)
    audit = {'study': study, 'model': model,
             'excluded_fit_participants': sorted(FIT_EXCLUDED[study]),
             'fitted_participants': int(observed.subj_idx.nunique()),
             'fitted_trials': len(observed), 'rt_valid_trials': len(rt),
             'dwell_valid_trials': len(dw), 'dwell_missing_bins': missing_dw,
             'n_ppc_draws': len(draw_names), 'rt_filter_seconds': list(RT_RANGE),
             'rt_binning': 'dplyr ntile within participant and PPC draw',
             'dwell_binning': 'cut on type-7 quintiles of unique DTA, within participant',
             'empirical_RT_uncertainty': 'participant SEM',
             'empirical_dwell_uncertainty': 'participant t-based 95% CI',
             'ppc_uncertainty': '2.5-97.5% across simulated datasets'}
    (cache_dir / f'{prefix}_rstyle_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    temp = cache_manifest.with_suffix('.json.part')
    temp.write_text(json.dumps(signature, indent=2) + '\n')
    temp.replace(cache_manifest)
    print(f'REBUILT R-STYLE PPC CURVES: {study}/{model}, {len(draw_names)} saved draws', flush=True)
    return output


def _merge_rstyle_curves(original: pd.DataFrame, rebuilt: pd.DataFrame) -> pd.DataFrame:
    """Replace ONLY the two originally pooled quintile metrics."""
    return pd.concat([original.loc[~original.metric.isin(RSTYLE_METRICS)], rebuilt],
                     ignore_index=True, sort=False)


def make_tables(study: str, model: str, result_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest = _check_ppc_saved(study, model, result_dir)
    prefix = model_prefix(model, study)
    data_dir = result_dir / 'data'
    names = ['observed_metrics', 'observed_subject_metrics',
             'replicate_metrics', 'replicate_subject_metrics']
    loaded = {name: _read_csv(data_dir / f'{prefix}_{name}.csv') for name in names}
    actual_draws = loaded['replicate_metrics']['draw'].nunique()
    actual_subject_draws = loaded['replicate_subject_metrics']['draw'].nunique()
    if actual_draws != manifest['ppc_draws_generated'] or actual_subject_draws != actual_draws:
        raise ValueError('Saved PPC draw counts do not match completed generation manifest')
    for label in names:
        frame = loaded[label]
        if label.startswith('replicate') and frame.duplicated(
                ['draw'] + (['metric', 'group', 'quintile'] if label == 'replicate_metrics'
                            else ['subj_idx', 'metric', 'group'])).any():
            raise ValueError(f'Duplicate PPC summary identity in {label}')
    core, curves = summarise_metrics(loaded['observed_metrics'], loaded['replicate_metrics'])
    # All PPC model summaries must reflect their fitted participant sample.
    check_path = data_dir / 'trials' / f'{prefix}_observed_trials.csv.gz'
    if check_path.is_file():
        fit_ids = pd.read_csv(check_path, usecols=['sub_id'])['sub_id']
        invalid = sorted(set(pd.to_numeric(fit_ids, errors='raise').astype(int))
                         & FIT_EXCLUDED[study])
        if invalid:
            raise ValueError(f'Excluded fit participants unexpectedly present in {model}: {invalid}')
    if model in IDENTITY_MODELS:
        curves = _merge_rstyle_curves(curves, rstyle_curves_from_archives(
            study, model, result_dir, manifest))
    boxes = subject_boxplot_data(loaded['observed_subject_metrics'],
                                 loaded['replicate_subject_metrics'], model, study)
    core.insert(0, 'study', study)
    core.insert(1, 'model', model)
    curves.insert(0, 'study', study)
    curves.insert(1, 'model', model)
    written = {
        f'{prefix}_ppc_core_summary.csv': core,
        f'{prefix}_ppc_quintile_curves.csv': curves,
        f'{prefix}_ppc_subject_boxplot_values.csv': boxes,
    }
    for name, frame in written.items():
        dest = result_dir / name
        tmp = dest.with_suffix('.csv.part')
        frame.to_csv(tmp, index=False, float_format='%.7f')
        tmp.replace(dest)
    return core, curves, boxes


def load_tables_or_rebuild(study: str, model: str, result_dir: Path):
    _check_ppc_saved(study, model, result_dir)
    prefix = model_prefix(model, study)
    raw_file = result_dir / 'data' / f'{prefix}_replicate_metrics.csv'
    if raw_file.is_file():
        return make_tables(study, model, result_dir)
    names = [f'{prefix}_ppc_core_summary.csv',
             f'{prefix}_ppc_quintile_curves.csv',
             f'{prefix}_ppc_subject_boxplot_values.csv']
    if not all((result_dir / name).is_file() for name in names):
        raise FileNotFoundError(f'No saved PPC data or complete legacy summaries in {result_dir}')
    if model in IDENTITY_MODELS:
        raise FileNotFoundError('R-style PPC requires saved trial archives and original per-draw metrics; '
                                'legacy-only summary tables cannot be rebinned.')
    print(f'REPLOT legacy PPC summaries {study}/{model}; '
          'new generation saves richer per-draw data for rebuilding tables.', flush=True)
    return tuple(_read_csv(result_dir / name) for name in names)


def run_plotting(study: str, model: str, root: Path = ROOT) -> Path:
    result_dir = root / 'studies' / study / 'results' / 'ppc' / model
    core, curves, boxes = load_tables_or_rebuild(study, model, result_dir)
    prefix = model_prefix(model, study)
    if model in IDENTITY_MODELS:
        if {'empirical_ci_low', 'empirical_ci_high'}.issubset(curves.columns):
            cirows = curves.loc[curves.metric.eq('p_choose_s_by_dwell_quintile')].copy()
            cirows.to_csv(result_dir / f'{prefix}_ppc_dwell_ci95_plot_summary.csv', index=False)
    fig_dir = result_dir / 'plots'
    plot_boxplots(boxes, study, model, fig_dir / f'{prefix}_ppc_mean_rt', 'mean_rt')
    plot_boxplots(boxes, study, model, fig_dir / f'{prefix}_ppc_mean_accuracy', 'accuracy')
    plots = ['mean_rt', 'mean_accuracy']
    if model in IDENTITY_MODELS:
        plot_choice_rt(curves, study, model, fig_dir / f'{prefix}_ppc_choice_rt_quintile')
        plot_dwell(curves, study, model, fig_dir / f'{prefix}_ppc_dwell_quintile')
        plots.extend(['choice_rt_quintile', 'dwell_quintile'])
    summary = {
        'study': study, 'model': model, 'plots_generated': plots,
        'tables_generated': ['core_summary', 'quintile_curves', 'subject_boxplot_values'],
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source': 'saved PPC draws; R-style RT and unique-DTA quintiles rebuilt offline',
    }
    (result_dir / f'{prefix}_ppc_plot_run.json').write_text(
        json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(f'PASS PPC PLOTS {study}/{model}: {len(plots)} figures, 3 summary tables;', flush=True)
    return result_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description='Generate PPC tables and figures')
    parser.add_argument('--study', choices=['study1', 'study2'], required=True)
    parser.add_argument('--model', choices=[*load_models(), 'all'], required=True)
    args = parser.parse_args(argv)
    names = list(load_models()) if args.model == 'all' else [args.model]
    for name in names:
        run_plotting(args.study, name)


if __name__ == '__main__':
    main()
