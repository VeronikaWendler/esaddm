#!/usr/bin/env python3
"""Plot ESaDDM+z parameter recovery for study1 and/or study2.

Reads existing rep_00 ... rep_19 recovery_parameters.csv and complete.json files.

Plotting:
- plots fitted parameters and derived coefficients with recovered posterior mean;
- plots derived ratios (theta_e, theta_s, delta_theta_s_minus_e) with posterior median;

"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DIRECT = ('a', 't', 'z', 'b_a', 'b_i', 'delta_a', 'delta_i')
DERIVED = ('b_ae', 'b_as', 'b_ie', 'b_is', 'theta_e', 'theta_s', 'delta_theta_s_minus_e')
RATIOS = ('theta_e', 'theta_s', 'delta_theta_s_minus_e')
PARAMETERS = DIRECT + DERIVED
STUDIES = ('study1', 'study2')
SUBJECTS = {'study1': 21, 'study2': 20}
COLORS = {'study1': '#39A991', 'study2': '#CF6C45'}
LABELS = {
    'a': r'$a$', 't': r'$t$', 'z': r'$z$',
    'b_a': r'$b_A$', 'b_i': r'$b_I$',
    'delta_a': r'$\delta_A$', 'delta_i': r'$\delta_I$',
    'b_ae': r'$b_{AE}$', 'b_as': r'$b_{AS}$',
    'b_ie': r'$b_{IE}$', 'b_is': r'$b_{IS}$',
    'theta_e': r'$\theta_E$', 'theta_s': r'$\theta_S$',
    'delta_theta_s_minus_e': r'$\theta_S-\theta_E$',
}
REQUIRED = {
    'study', 'rep', 'scope', 'subj_idx', 'parameter', 'true',
    'recovered_mean', 'recovered_median', 'hdi_lower', 'hdi_upper',
    'covered_95', 'r_hat', 'ess_bulk', 'ess_tail',
}

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 13,
    'axes.titlesize': 17, 'axes.labelsize': 14,
    'xtick.labelsize': 12, 'ytick.labelsize': 12,
    'figure.titlesize': 19, 'legend.fontsize': 12,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})


def values(series):
    return pd.to_numeric(series, errors='coerce').to_numpy(dtype=float)


def load_study(root, study, reps, allow_incomplete):
    base = root / 'studies' / study / 'results' / 'recovery' / 'esaddm_z'
    frames, audits = [], []
    for rep in range(reps):
        folder = base / f'rep_{rep:02d}'
        csv, manifest = folder / 'recovery_parameters.csv', folder / 'complete.json'
        if not csv.is_file() or not manifest.is_file():
            if allow_incomplete:
                warnings.warn(f'missing replicate: {folder}')
                continue
            raise FileNotFoundError(f'missing recovery results: {folder}')
        meta = json.loads(manifest.read_text(encoding='utf-8'))
        if meta.get('status') != 'completed':
            if allow_incomplete:
                warnings.warn(f'incomplete replicate: {folder}')
                continue
            raise ValueError(f'incomplete recovery manifest: {manifest}')
        frame = pd.read_csv(csv)
        missing = REQUIRED - set(frame.columns)
        if missing:
            raise ValueError(f'{csv}: missing columns {sorted(missing)}')
        if (frame.empty or frame.study.nunique() != 1 or frame.study.iloc[0] != study
                or frame.rep.nunique() != 1 or int(frame.rep.iloc[0]) != rep):
            raise ValueError(f'{csv}: inconsistent study or replication number')
        if frame.duplicated(['scope', 'subj_idx', 'parameter']).any():
            raise ValueError(f'{csv}: duplicated parameter rows')
        for scope, count in (('group', 1), ('individual', SUBJECTS[study])):
            for param in PARAMETERS:
                n = (frame.scope.eq(scope) & frame.parameter.eq(param)).sum()
                if n != count:
                    raise ValueError(f'{csv}: expected {count} entries for {scope}/{param}, found {n}')
        frame['ratio_warning'] = frame.get('ratio_warning', pd.Series('', index=frame.index))
        frame['study'], frame['rep'] = study, rep
        direct = frame[frame.parameter.isin(DIRECT)]
        converged = (pd.to_numeric(direct.r_hat, errors='coerce').le(1.01)
                     & pd.to_numeric(direct.ess_bulk, errors='coerce').ge(400)
                     & pd.to_numeric(direct.ess_tail, errors='coerce').ge(400)).all()
        audits.append({'study': study, 'rep': rep, 'converged': bool(converged),
                       'max_rhat': values(direct.r_hat).max(),
                       'min_ess_bulk': values(direct.ess_bulk).min(),
                       'min_ess_tail': values(direct.ess_tail).min()})
        frames.append(frame)
    if not frames:
        raise ValueError(f'no completed replications: {base}')
    return pd.concat(frames, ignore_index=True), pd.DataFrame(audits)


def estimator(param, scope='individual'):
    return 'median' if scope == 'individual' and param in RATIOS else 'mean'


def recovery_stats(part, param, scope):
    x, y = values(part.true), values(part[f'recovered_{estimator(param, scope)}'])
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    r = (float(np.corrcoef(x, y)[0, 1]) if len(x) >= 3
         and np.std(x) > 1e-12 and np.std(y) > 1e-12 else np.nan)
    cov = part.covered_95.astype('string').str.lower().map({'true': 1, 'false': 0, '1': 1, '0': 0})
    coverage = float(cov.mean()) if cov.notna().any() else np.nan
    return dict(n=len(x), r=r, r2=r*r if np.isfinite(r) else np.nan,
                bias=float(np.mean(y-x)) if len(x) else np.nan,
                rmse=float(np.sqrt(np.mean((y-x)**2))) if len(x) else np.nan,
                coverage95=coverage)


def summary_table(df, study):
    rows = []
    for scope in ('individual', 'group'):
        for param in PARAMETERS:
            subset = df[df.scope.eq(scope) & df.parameter.eq(param)]
            rows.append({'study': study, 'scope': scope, 'parameter': param,
                         'estimator': estimator(param, scope),
                         **recovery_stats(subset, param, scope)})
    return pd.DataFrame(rows)


def mark_display_outliers(combined):
    """Hide only extreme individual ratio values, using standard Tukey outer fences.

    Values outside Q1 - 3*IQR or Q3 + 3*IQR are extreme boxplot outliers.
    Limits are computed once, pooling both studies, to keep their plots comparable.
    No observations are removed from recovery statistics or stored data.
    """
    df = combined.copy()
    df['plot_keep'] = True
    df['plot_note'] = ''
    for param in RATIOS:
        mask = df.scope.eq('individual') & df.parameter.eq(param)
        part = df.loc[mask]
        x, y = values(part.true), values(part.recovered_median)
        pooled = np.r_[x, y]
        pooled = pooled[np.isfinite(pooled)]
        if len(pooled) < 4:
            continue
        q1, q3 = np.percentile(pooled, [25, 75])
        iqr = q3 - q1
        if iqr <= 0:
            continue
        lower, upper = q1 - 3*iqr, q3 + 3*iqr
        invalid = ~np.isfinite(x) | ~np.isfinite(y)
        extreme = (x < lower) | (x > upper) | (y < lower) | (y > upper)
        reason = np.where(invalid, 'nonfinite ratio', np.where(extreme, 'extreme ratio outlier', ''))
        df.loc[mask, 'plot_keep'] = reason == ''
        df.loc[mask, 'plot_note'] = reason
    return df


def limits_for_plots(combined):
    """Use matching limits between study1/study2 for each parameter."""
    output = {'individual': {}, 'group': {}}
    for scope in ('individual', 'group'):
        for param in PARAMETERS:
            part = combined[combined.scope.eq(scope) & combined.parameter.eq(param)]
            if scope == 'individual':
                part = part[part.plot_keep]
                vals = np.r_[values(part.true), values(part[f'recovered_{estimator(param, scope)}'])]
            else:
                vals = np.r_[values(part.true), values(part.recovered_mean),
                             values(part.hdi_lower), values(part.hdi_upper)]
            vals = vals[np.isfinite(vals)]
            if len(vals):
                low, high = float(vals.min()), float(vals.max())
                pad = max((high - low)*.06, .012*max(abs(low), abs(high), 1))
                output[scope][param] = (low-pad, high+pad)
            else:
                output[scope][param] = (-1.0, 1.0)
    return output


def r2_label(ax, value):
    label = f'{value:.2f}' if np.isfinite(value) else 'n/a'
    ax.text(.045, .955, rf'$R^2 = {label}$', transform=ax.transAxes,
            ha='left', va='top', fontsize=13,
            bbox=dict(facecolor='white', edgecolor='#dddddd',
                      boxstyle='round,pad=.18', alpha=.92))


def draw_panel(ax, df, stats, scope, param, lim, color):
    part = df[df.scope.eq(scope) & df.parameter.eq(param)]
    if scope == 'individual':
        part = part[part.plot_keep]
        x, y = values(part.true), values(part[f'recovered_{estimator(param, scope)}'])
        good = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[good], y[good], s=20, color=color,
                   alpha=.42, edgecolors='none', rasterized=True)
        ax.plot(lim, lim, '--', color='#333333', lw=1.2)
        ax.set(xlim=lim, ylim=lim, xlabel='generating value',
               ylabel='recovered posterior ' + estimator(param, scope))
        ax.set_aspect('equal', adjustable='box')
    else:
        part = part.sort_values('rep')
        rep = values(part.rep)
        mean, truth = values(part.recovered_mean), values(part.true)
        lower, upper = values(part.hdi_lower), values(part.hdi_upper)
        ok = np.isfinite(lower) & np.isfinite(upper)
        ax.vlines(rep[ok], lower[ok], upper[ok], color='#999999', lw=2)
        ax.scatter(rep[np.isfinite(mean)], mean[np.isfinite(mean)],
                   s=35, color=color, label='recovered mean', zorder=4)
        ax.scatter(rep[np.isfinite(truth)], truth[np.isfinite(truth)],
                   s=35, color='black', marker='x', label='generating value', zorder=5)
        ax.set(xlim=(rep.min()-.7, rep.max()+.7), ylim=lim,
               xlabel='recovery replicate', ylabel='parameter value')
        ax.set_xticks(rep[::max(1, len(rep)//6)].astype(int))
    ax.set_title(LABELS[param], pad=8)
    r2_label(ax, stats.r2)


def save(fig, folder, stem, dpi):
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f'{stem}.pdf', bbox_inches='tight', facecolor='white')
    fig.savefig(folder / f'{stem}.png', dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def figures(df, summary, limits, folder, study, dpi):
    color = COLORS[study]
    for scope in ('individual', 'group'):
        for category, params in (('fitted', DIRECT), ('derived', DERIVED)):
            fig, axes = plt.subplots(2, 4, figsize=(19, 10.5), layout='constrained')
            for ax, param in zip(axes.flat, params):
                stats = summary[summary.scope.eq(scope) & summary.parameter.eq(param)].iloc[0]
                draw_panel(ax, df, stats, scope, param, limits[scope][param], color)
            axes.flat[-1].axis('off')
            if scope == 'group':
                handles, labels = axes.flat[0].get_legend_handles_labels()
                fig.legend(handles, labels, loc='lower right', bbox_to_anchor=(.98, .04), frameon=False)
            if scope == 'individual' and category == 'derived':
                n_hidden = int((df.scope.eq('individual') & df.parameter.isin(RATIOS) & ~df.plot_keep).sum())
                if n_hidden:
                    axes.flat[-1].text(.02, .9, f'extreme ratio outliers not shown: {n_hidden}\n'
                                                   'r² includes all observations',
                                        transform=axes.flat[-1].transAxes,
                                        ha='left', va='top', fontsize=12, color='#666666')
            fig.suptitle(f'{study} | {scope}-level recovery | {category}')
            save(fig, folder, f'{"participant" if scope == "individual" else "group"}_recovery_{category}', dpi)

        single = folder / ('participant_individual' if scope == 'individual' else 'group_individual')
        for param in PARAMETERS:
            fig, ax = plt.subplots(figsize=(7.2, 6.3), layout='constrained')
            stats = summary[summary.scope.eq(scope) & summary.parameter.eq(param)].iloc[0]
            draw_panel(ax, df, stats, scope, param, limits[scope][param], color)
            if scope == 'group':
                ax.legend(loc='lower right', frameon=False)
            fig.suptitle(f'{study} | {scope}-level recovery', fontsize=18)
            save(fig, single, f'{scope}_{param}', dpi)


def write_short_summary(folder, study, summary, audit, omitted):
    fitted = summary[summary.scope.eq('individual') & summary.parameter.isin(DIRECT)]
    lines = [f'{study}: ESaDDM+z parameter recovery',
             f'replications: {len(audit)}; fully converged: {int(audit.converged.sum())}/{len(audit)}',
             f'fitted parameters: r range {fitted.r.min():.2f}–{fitted.r.max():.2f}; '
             f'median r = {fitted.r.median():.2f}',
             '', 'individual-level recovery (main results):',
             'parameter             r      r²     rmse    95% coverage']
    for _, row in summary[summary.scope.eq('individual')].iterrows():
        lines.append(f'{row.parameter:<22} {row.r:>5.2f}  {row.r2:>6.2f}'
                     f'  {row.rmse:>7.3f}    {row.coverage95:>6.1%}')
    lines += ['', f'extreme ratio outliers omitted from figures: {len(omitted)}',
              'outliers: conventional Tukey extreme-outlier rule (beyond 3 × IQR);',
              'applied only to individual-level ratio plots.',
              'all observations (including outliers) contribute to recovery statistics.',
              'fitted coefficients use posterior means; individual derived ratios use posterior medians.',
              'group recovery statistics and bias are available in recovery_summary.csv.',
              'r² is squared Pearson r (descriptive recovery).',
              '95% coverage uses the saved posterior HDIs.']
    (folder / 'recovery_summary_for_paper.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (folder / 'README_RESULTS.txt').write_text(
        f'{study}: {len(audit)} completed recovery replicates\n'
        'plots/: fitted and derived recovery in PDF and PNG\n'
        'recovery_summary.csv: r, r², bias, RMSE, 95% coverage\n'
        'recovery_summary_for_paper.txt: short individual-level summary\n'
        'rep_convergence_audit.csv: convergence by replicate\n'
        'display_omitted_points.csv: any outliers omitted from figures only\n', encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path.cwd())
    p.add_argument('--study', choices=('study1', 'study2', 'both'), default='both')
    p.add_argument('--expected-reps', type=int, default=20)
    p.add_argument('--allow-incomplete', action='store_true')
    p.add_argument('--out', type=Path, default=None)
    p.add_argument('--dpi', type=int, default=200)
    args = p.parse_args()
    if args.expected_reps < 1:
        p.error('--expected-reps must be positive')
    root = args.root.resolve()
    data, audits = {}, {}
    for study in STUDIES:
        data[study], audits[study] = load_study(root, study, args.expected_reps, args.allow_incomplete)
    combined = mark_display_outliers(pd.concat([data[s] for s in STUDIES], ignore_index=True))
    limits = limits_for_plots(combined)
    selected = STUDIES if args.study == 'both' else (args.study,)
    for study in selected:
        df = combined[combined.study.eq(study)].copy()
        summary = summary_table(df, study)
        omitted = df[df.scope.eq('individual') & ~df.plot_keep]
        folder = (args.out.resolve() / study if args.out else
                  root / 'studies' / study / 'results' / 'recovery' / 'esaddm_z' / 'summary')
        folder.mkdir(parents=True, exist_ok=True)
        summary.to_csv(folder / 'recovery_summary.csv', index=False)
        audits[study].to_csv(folder / 'rep_convergence_audit.csv', index=False)
        # Export only omitted rows; original full tables remain in rep_XX directories.
        omitted[['study', 'rep', 'subj_idx', 'parameter', 'true', 'recovered_mean',
                 'recovered_median', 'ratio_warning', 'plot_note']].to_csv(
                     folder / 'display_omitted_points.csv', index=False)
        write_short_summary(folder, study, summary, audits[study], omitted)
        figures(df, summary, limits, folder / 'plots', study, args.dpi)
        print(f'{study}: {len(audits[study])} recovery replicates, '
              f'{len(omitted)} ratio outliers omitted from display only; figures: {folder / "plots"}', flush=True)


if __name__ == '__main__':
    main()
