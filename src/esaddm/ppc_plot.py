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
import numpy as np
import pandas as pd

from .models import ROOT, load_models
from .ppc import (
    OBS_E, OBS_S, PPC_E, PPC_S, OBS_OTHER, PPC_OTHER,
    FONT_TITLE, FONT_PANEL, FONT_AXIS, FONT_TICKS, QUINTILES,
    IDENTITY_MODELS, model_prefix, summarise_metrics, subject_boxplot_data,
)

def style_plot(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.4)
    ax.spines['bottom'].set_linewidth(1.4)
    ax.tick_params(labelsize=FONT_TICKS, width=1.2)
    ax.grid(False)


def save_plot(fig, output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix('.png'), dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_boxplots(subject_data: pd.DataFrame, study: str, model: str, output: Path,
                  metric: str):
    data = subject_data.loc[subject_data.metric.eq(metric)].copy()
    if model == 'accuracy_ee':
        groups = ['Overall'] if metric == 'accuracy' else ['Correct', 'Error']
    else:
        groups = ['E', 'S']
    fig, ax = plt.subplots(figsize=(12.5, 8.1))
    for i, group in enumerate(groups):
        sub = data.loc[data.group.eq(group)].sort_values('subj_idx')
        if sub.empty:
            raise ValueError(f'Missing {metric} group {group}')
        emp_color = {'E':OBS_E, 'S':OBS_S}.get(group, OBS_OTHER)
        pred_color = {'E':PPC_E, 'S':PPC_S}.get(group, PPC_OTHER)
        for offset, name, color in [(-.18, 'empirical_value', emp_color),
                                    (.18, 'ppc_subject_mean', pred_color)]:
            x = i + offset
            bp = ax.boxplot(sub[name].to_numpy(), positions=[x], widths=.27,
                            patch_artist=True, showfliers=False,
                            medianprops={'color':'#252525','linewidth':2.4},
                            whiskerprops={'color':'#444444','linewidth':1.2},
                            capprops={'color':'#444444','linewidth':1.2})
            bp['boxes'][0].set_facecolor(color)
            bp['boxes'][0].set_alpha(.78)
            bp['boxes'][0].set_edgecolor('#282828')
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, fontsize=FONT_AXIS)
    ax.set_ylabel('Mean RT (seconds)' if metric == 'mean_rt' else 'Mean accuracy (0–1)',
                  fontsize=FONT_AXIS)
    if metric == 'accuracy':
        ax.set_ylim(0, 1.05)
    ax.set_title(('Mean RT' if metric == 'mean_rt' else 'Mean accuracy') +
                 f' — {model} ({study})', fontsize=FONT_TITLE, pad=18)
    import matplotlib.patches as mpatches
    if model == 'accuracy_ee':
        legend_items = [mpatches.Patch(facecolor=OBS_OTHER, label='Empirical'),
                        mpatches.Patch(facecolor=PPC_OTHER, label='PPC')]
    else:
        legend_items = [mpatches.Patch(facecolor=OBS_E, label='Empirical E'),
                        mpatches.Patch(facecolor=PPC_E, label='PPC E'),
                        mpatches.Patch(facecolor=OBS_S, label='Empirical S'),
                        mpatches.Patch(facecolor=PPC_S, label='PPC S')]
    ax.legend(handles=legend_items, fontsize=15, frameon=False,
              ncol=2, loc='best')
    style_plot(ax)
    fig.tight_layout()
    save_plot(fig, output)


def _curve_slice(curves, metric):
    part = curves.loc[curves.metric.eq(metric)].sort_values('quintile')
    if len(part)!=5 or not np.array_equal(part.quintile.to_numpy(), np.array(QUINTILES)):
        raise ValueError(f'Incomplete quintile PPC curve for {metric}')
    return part


def plot_choice_rt(curves: pd.DataFrame, study: str, model: str, output: Path):
    rows = _curve_slice(curves, 'p_choose_s_by_rt_quintile')
    x = np.asarray(QUINTILES, dtype=float)
    fig, ax = plt.subplots(figsize=(12.5, 8.0))
    for choice, shift, obs_color, pred_color in (
        ('E', -.12, OBS_E, PPC_E), ('S', +.12, OBS_S, PPC_S)
    ):
        empirical = rows.empirical.to_numpy(float) if choice == 'S' else 1-rows.empirical.to_numpy(float)
        ppc = rows.ppc_mean.to_numpy(float) if choice == 'S' else 1-rows.ppc_mean.to_numpy(float)
        low = rows.ppc_pi95_low.to_numpy(float) if choice == 'S' else 1-rows.ppc_pi95_high.to_numpy(float)
        high = rows.ppc_pi95_high.to_numpy(float) if choice == 'S' else 1-rows.ppc_pi95_low.to_numpy(float)
        ax.bar(x + shift, empirical, width=.23, alpha=.76, color=obs_color,
               edgecolor='#222222', linewidth=1.0, label=f'Empirical {choice}', zorder=1)
        ax.plot(x+shift, ppc, color=pred_color, marker='o', linewidth=3.5,
                markersize=10, label=f'PPC {choice}', zorder=4)
        ax.errorbar(x+shift, ppc, yerr=[ppc-low, high-ppc], fmt='none',
                    ecolor=pred_color, elinewidth=2.2, capsize=5, zorder=3)
    ax.set_xticks(x)
    ax.set_xlabel('RT quintile (fastest → slowest)', fontsize=FONT_AXIS)
    ax.set_ylabel('Choice probability', fontsize=FONT_AXIS)
    ax.set_ylim(0, 1.0)
    ax.set_title(f'Choice by RT quintile — {model} ({study})', fontsize=FONT_TITLE, pad=18)
    ax.legend(frameon=False, fontsize=15, ncol=2)
    style_plot(ax)
    fig.tight_layout()
    save_plot(fig, output)


def plot_dwell(curves: pd.DataFrame, study: str, model: str, output: Path):
    rows = _curve_slice(curves, 'p_choose_s_by_dwell_quintile')
    x = np.asarray(QUINTILES, dtype=float)
    fig, ax = plt.subplots(figsize=(12.5, 8.0))
    ax.bar(x, rows.empirical.to_numpy(float), width=.72, facecolor=OBS_S, alpha=.76,
           edgecolor='#222222', linewidth=1.0, label='Empirical P(S)', zorder=1)
    ax.plot(x, rows.ppc_mean.to_numpy(float), color=PPC_S, marker='o', linewidth=3.4,
            markersize=10, label='PPC P(S)', zorder=4)
    pred = rows.ppc_mean.to_numpy(float)
    ax.errorbar(x, pred, yerr=[pred - rows.ppc_pi95_low.to_numpy(float),
                              rows.ppc_pi95_high.to_numpy(float) - pred],
                fmt='none', ecolor=PPC_S, elinewidth=2.1, capsize=5, zorder=3)
    ax.set_xticks(x)
    # Rank quintiles do not guarantee that the middle bin is truly zero.
    ax.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], fontsize=FONT_TICKS)
    ax.set_xlabel('Dwell-time advantage quintile (S − E, ms)', fontsize=FONT_AXIS)
    ax.set_ylabel('P(choose S)', fontsize=FONT_AXIS)
    ax.set_ylim(0, 1)
    ax.set_title(f'Choice by dwell-time advantage — {model} ({study})',
                 fontsize=FONT_TITLE, pad=18)
    ax.legend(frameon=False, fontsize=17)
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
    """Recompute tables without rerunning any HDDM simulator."""
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
    # Retain established filenames for compatibility with cross-model plots.
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
    """Prefer reproducible per-draw data, but allow old completed smoke PPC outputs.

    Historical PPC runs saved final tables but not the per-draw participant-level
    data needed to regenerate them. They can still be *replotted*, though.
    """
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
    """Lightweight, rerunnable postprocessing; never loads/changes fitted models."""
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
    print(f'PASS PPC PLOTS {study}/{model}: {len(plots)} figures, 3 summary tables; '
          , flush=True)
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
