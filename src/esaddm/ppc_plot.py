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
import numpy as np
import pandas as pd

from .models import ROOT, load_models
from .ppc import (
    OBS_E, OBS_S, PPC_E, PPC_S, OBS_OTHER, PPC_OTHER,
    QUINTILES, IDENTITY_MODELS, model_prefix, summarise_metrics, subject_boxplot_data,
)

# Slightly larger typography throughout.
FONT_TITLE = 24
FONT_PANEL = 20
FONT_AXIS = 19
FONT_TICKS = 16
FONT_LEGEND = 16

# Stronger colours for empirical bars/lines.
EMP_E = '#19B5E5'
EMP_S = '#A13AD8'
PRED_E = PPC_E
PRED_S = PPC_S
EDGE = '#000000'
MIDLINE = '#111111'

RT_LABELS = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
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

    fig, ax = plt.subplots(figsize=(10.6, 7.6))
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
    ax.set_title(('Mean RT' if metric == 'mean_rt' else 'Mean accuracy') +
                 f' — {model} ({study})', fontsize=FONT_TITLE, pad=16)

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
              ncol=2, loc='best')
    style_plot(ax)
    fig.tight_layout()
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

    fig, ax = plt.subplots(figsize=(10.8, 7.6))
    ax.bar(x - width / 2.0, empirical_e, width=width, color=EMP_E,
           edgecolor=EDGE, linewidth=1.2, alpha=.96, zorder=1)
    ax.bar(x + width / 2.0, empirical_s, width=width, color=EMP_S,
           edgecolor=EDGE, linewidth=1.2, alpha=.96, zorder=1)

    ax.plot(x - width / 2.0, ppc_e, color=PRED_E, marker='o', linewidth=3.0,
            markersize=8.5, label='PPC E', zorder=4)
    ax.errorbar(x - width / 2.0, ppc_e, yerr=[ppc_e - low_e, high_e - ppc_e],
                fmt='none', ecolor=PRED_E, elinewidth=2.0, capsize=5, zorder=4)
    ax.plot(x + width / 2.0, ppc_s, color=PRED_S, marker='o', linewidth=3.0,
            markersize=8.5, label='PPC S', zorder=4)
    ax.errorbar(x + width / 2.0, ppc_s, yerr=[ppc_s - low_s, high_s - ppc_s],
                fmt='none', ecolor=PRED_S, elinewidth=2.0, capsize=5, zorder=4)

    ax.axhline(50, color=MIDLINE, linestyle=(0, (1.2, 4.0)), linewidth=2.0, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(RT_LABELS, fontsize=FONT_TICKS)
    ax.set_xlabel('RT quintile (shortest to longest)', fontsize=FONT_AXIS)
    ax.set_ylabel('Probability (%)', fontsize=FONT_AXIS)
    ax.set_title(f'E vs S choice by RT quintile — {model} ({study})',
                 fontsize=FONT_TITLE, pad=16)
    _probability_limits(ax, [empirical_e, empirical_s, low_e, high_e, low_s, high_s],
                        scale='percent', include_half=True)

    legend_handles = [
        mpatches.Patch(facecolor=EMP_E, edgecolor=EDGE, label='Empirical E'),
        mpatches.Patch(facecolor=EMP_S, edgecolor=EDGE, label='Empirical S'),
        Line2D([0], [0], color=PRED_E, marker='o', linewidth=3.0, markersize=8.0, label='PPC E'),
        Line2D([0], [0], color=PRED_S, marker='o', linewidth=3.0, markersize=8.0, label='PPC S'),
    ]
    ax.legend(handles=legend_handles, title='Chosen option', title_fontsize=FONT_LEGEND + 2,
              fontsize=FONT_LEGEND, frameon=False, ncol=2, loc='best')
    style_plot(ax)
    fig.tight_layout()
    save_plot(fig, output)


def plot_dwell(curves: pd.DataFrame, study: str, model: str, output: Path):
    rows = _curve_slice(curves, 'p_choose_s_by_dwell_quintile')
    x = np.arange(1, 6, dtype=float)
    empirical = rows.empirical.to_numpy(float)
    pred = rows.ppc_mean.to_numpy(float)
    pred_low = rows.ppc_pi95_low.to_numpy(float)
    pred_high = rows.ppc_pi95_high.to_numpy(float)
    emp_low, emp_high = _empirical_error_columns(rows)

    fig, ax = plt.subplots(figsize=(9.8, 7.3))
    ax.axhline(0.5, color='#7a7a7a', linestyle=(0, (1.2, 3.2)), linewidth=1.6, zorder=0)

    # empirical line first
    ax.plot(x, empirical, color=EMP_S, marker='o', linewidth=3.2,
            markersize=9.0, label='Empirical', zorder=3)
    if emp_low is not None and emp_high is not None:
        ax.errorbar(x, empirical, yerr=[empirical - emp_low, emp_high - empirical],
                    fmt='none', ecolor=EMP_S, elinewidth=1.9, capsize=4, zorder=3)

    ax.plot(x, pred, color=PRED_S, marker='o', linewidth=3.0,
            markersize=8.5, label='PPC', zorder=4)
    ax.errorbar(x, pred, yerr=[pred - pred_low, pred_high - pred],
                fmt='none', ecolor=PRED_S, elinewidth=2.0, capsize=5, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(DWELL_LABELS, fontsize=FONT_TICKS)
    ax.set_xlabel('Dwell-Time Advantage (S > E)', fontsize=FONT_AXIS)
    ax.set_ylabel('P(Choose Symbolic)', fontsize=FONT_AXIS)
    ax.set_title(f'Choice by dwell-time advantage — {model} ({study})',
                 fontsize=FONT_TITLE, pad=16)
    _probability_limits(ax, [empirical, pred, pred_low, pred_high],
                        scale='proportion', include_half=True)
    ax.legend(frameon=False, fontsize=FONT_LEGEND, loc='best')
    style_plot(ax)
    fig.tight_layout()
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
    print(f'REPLOT legacy PPC summaries {study}/{model}; '
          'new generation saves richer per-draw data for rebuilding tables.', flush=True)
    return tuple(_read_csv(result_dir / name) for name in names)


def run_plotting(study: str, model: str, root: Path = ROOT) -> Path:
    result_dir = root / 'studies' / study / 'results' / 'ppc' / model
    core, curves, boxes = load_tables_or_rebuild(study, model, result_dir)
    prefix = model_prefix(model, study)
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
        'source': 'completed PPC generation files in data/',
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
