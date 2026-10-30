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

DIRECT = ['a', 't', 'z', 'b_a', 'b_i', 'delta_a', 'delta_i']
DERIVED = ['b_ae', 'b_as', 'b_ie', 'b_is', 'theta_e', 'theta_s', 'delta_theta_s_minus_e']
RATIOS = {'theta_e', 'theta_s', 'delta_theta_s_minus_e'}
PARAMETERS = DIRECT + DERIVED
STUDIES = ('study1', 'study2')
EXPECTED_SUBJECTS = {'study1': 21, 'study2': 20}
MODEL = 'esaddm_z'
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
    'font.family': 'DejaVu Sans',
    'font.size': 13,
    'axes.titlesize': 17,
    'axes.labelsize': 13.5,
    'xtick.labelsize': 11.5,
    'ytick.labelsize': 11.5,
    'figure.titlesize': 19,
    'legend.fontsize': 12,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# Conservative, deterministic display-only rule for posterior ratios.
# Exclude explicitly warned ratios and points beyond 8 robust standard deviations
# of the pooled generating+median values, with a common range across both studies.
# 1.4826 * MAD is the Gaussian-consistent robust spread estimate.
RATIO_MAD_MULTIPLIER = 8.0
COLORS = {'study1': "#39A991", 'study2': "#CF6C45"}


def numeric(values):
    return pd.to_numeric(values, errors='coerce').to_numpy(dtype=float)


def coverage_values(series):
    text_values = series.astype('string').str.strip().str.lower()
    text_values = text_values.mask(text_values.eq('true').fillna(False), '1')
    text_values = text_values.mask(text_values.eq('false').fillna(False), '0')
    return pd.to_numeric(text_values, errors='coerce')


def load_recovery(root: Path, study: str, reps: int, allow_incomplete: bool):
    frames, audits = [], []
    parent = root / 'studies' / study / 'results' / 'recovery' / MODEL
    for rep in range(reps):
        folder = parent / f'rep_{rep:02d}'
        csv_path, manifest = folder / 'recovery_parameters.csv', folder / 'complete.json'
        if not csv_path.is_file() or not manifest.is_file():
            if allow_incomplete:
                warnings.warn(f'missing replicate, not plotted: {folder}')
                continue
            raise FileNotFoundError(f'missing recovery_parameters.csv or complete.json: {folder}')
        meta = json.loads(manifest.read_text(encoding='utf-8'))
        if meta.get('status') != 'completed':
            if allow_incomplete:
                warnings.warn(f'incomplete replicate, not plotted: {folder}')
                continue
            raise ValueError(f'replicate not marked completed: {manifest}')
        frame = pd.read_csv(csv_path)
        missing = REQUIRED - set(frame.columns)
        if missing:
            raise ValueError(f'{csv_path}: missing columns {sorted(missing)}')
        if frame.empty or frame.study.nunique() != 1 or frame.study.iloc[0] != study:
            raise ValueError(f'{csv_path}: study mismatch')
        if frame.rep.nunique() != 1 or int(frame.rep.iloc[0]) != rep:
            raise ValueError(f'{csv_path}: replicate mismatch')
        if frame.duplicated(['scope', 'subj_idx', 'parameter']).any():
            raise ValueError(f'{csv_path}: duplicate parameter records')
        if set(frame.parameter.unique()) != set(PARAMETERS):
            raise ValueError(f'{csv_path}: parameter set does not match all 14 parameters')
        for scope, count in [('group', 1), ('individual', EXPECTED_SUBJECTS[study])]:
            for param in PARAMETERS:
                actual = int((frame.scope.eq(scope) & frame.parameter.eq(param)).sum())
                if actual != count:
                    raise ValueError(f'{csv_path}: expected {count} rows for {scope}/{param}, got {actual}')
        frame['study'] = study
        frame['rep'] = rep
        for col, default in [
            ('ratio_warning', ''),
            ('denominator_ci95_contains_zero', np.nan),
            ('fraction_finite_posterior_draws', np.nan),
        ]:
            if col not in frame:
                frame[col] = default
        direct = frame[frame.parameter.isin(DIRECT)]
        valid = (pd.to_numeric(direct.r_hat, errors='coerce').le(1.01)
                 & pd.to_numeric(direct.ess_bulk, errors='coerce').ge(400)
                 & pd.to_numeric(direct.ess_tail, errors='coerce').ge(400))
        audits.append({
            'study': study, 'rep': rep, 'n_rows': len(frame),
            'manifest_convergence_ok': meta.get('direct_convergence_ok'),
            'independently_checked_convergence_ok': bool(valid.all()),
            'max_direct_rhat': numeric(direct.r_hat).max(),
            'min_direct_ess_bulk': numeric(direct.ess_bulk).min(),
            'min_direct_ess_tail': numeric(direct.ess_tail).min(),
            'n_ratio_warning_rows': int(frame.ratio_warning.fillna('').astype(str).str.len().gt(0).sum()),
        })
        frames.append(frame)
    if not frames:
        raise ValueError(f'no completed recovery replicates for {study} in {parent}')
    return pd.concat(frames, ignore_index=True), pd.DataFrame(audits)


def preferred_estimator(scope: str, param: str) -> str:
    # Ratio posterior means may be undefined/unstable when attended denominators
    # approach zero; use the median for plotted and headline ratio summaries.
    return 'median' if param in RATIOS else 'mean'


def recovery_statistics(part: pd.DataFrame, estimator: str) -> dict:
    true = numeric(part['true'])
    recovered = numeric(part[f'recovered_{estimator}'])
    valid = np.isfinite(true) & np.isfinite(recovered)
    x, y = true[valid], recovered[valid]
    r = float(np.corrcoef(x, y)[0, 1]) if (len(x) >= 3 and
        np.std(x) > 1e-12 and np.std(y) > 1e-12) else np.nan
    slope = float(np.polyfit(x, y, 1)[0]) if len(x) >= 2 and np.std(x) > 1e-12 else np.nan
    within_rep_r = []
    for _, g in part.groupby('rep'):
        gx, gy = numeric(g.true), numeric(g[f'recovered_{estimator}'])
        ok = np.isfinite(gx) & np.isfinite(gy)
        if ok.sum() >= 3 and np.std(gx[ok]) > 1e-12 and np.std(gy[ok]) > 1e-12:
            within_rep_r.append(float(np.corrcoef(gx[ok], gy[ok])[0, 1]))
    covered = coverage_values(part.covered_95)
    return {
        'n': int(len(x)), 'n_reps': int(part.rep.nunique()),
        'r': r, 'r2': float(r*r) if np.isfinite(r) else np.nan,
        'median_within_rep_r': float(np.median(within_rep_r)) if within_rep_r else np.nan,
        'slope': slope,
        'bias': float(np.mean(y - x)) if len(x) else np.nan,
        'rmse': float(np.sqrt(np.mean((y - x)**2))) if len(x) else np.nan,
        'mae': float(np.mean(np.abs(y - x))) if len(x) else np.nan,
        'coverage95': float(covered.mean()) if covered.notna().any() else np.nan,
        'coverage_n': int(covered.notna().sum()),
        'median_rhat': float(pd.to_numeric(part.r_hat, errors='coerce').median()),
        'max_rhat': float(pd.to_numeric(part.r_hat, errors='coerce').max()),
        'min_bulk_ess': float(pd.to_numeric(part.ess_bulk, errors='coerce').min()),
        'min_tail_ess': float(pd.to_numeric(part.ess_tail, errors='coerce').min()),
        'n_ratio_warnings': int(part.ratio_warning.fillna('').astype(str).str.len().gt(0).sum()),
    }


def make_summaries(df: pd.DataFrame, study: str):
    detailed, primary = [], []
    for scope in ('group', 'individual'):
        for param in PARAMETERS:
            part = df[df.scope.eq(scope) & df.parameter.eq(param)]
            for estimator in ('mean', 'median'):
                row = {
                    'study': study, 'scope': scope, 'parameter': param,
                    'kind': ('direct' if param in DIRECT else
                             'derived_ratio' if param in RATIOS else 'derived_coefficient'),
                    'estimator': estimator,
                    **recovery_statistics(part, estimator),
                }
                detailed.append(row)
                if estimator == preferred_estimator(scope, param):
                    primary.append(row.copy())
    return pd.DataFrame(primary), pd.DataFrame(detailed)


def padded_limits(data):
    z = np.asarray(data, dtype=float)
    z = z[np.isfinite(z)]
    if not len(z):
        return (-1.0, 1.0)
    lo, hi = float(z.min()), float(z.max())
    pad = max((hi - lo) * 0.055, 0.012 * max(abs(lo), abs(hi), 1))
    return lo - pad, hi + pad


def ratio_display_rules(combined: pd.DataFrame):
    """Calculate cross-study plot-only limits and exclusion rules for ratios."""
    rules = {}
    for param in RATIOS:
        part = combined[combined.scope.eq('individual') & combined.parameter.eq(param)]
        vals = np.r_[numeric(part.true), numeric(part.recovered_median)]
        vals = vals[np.isfinite(vals)]
        if not len(vals):
            rules[param] = (-1.0, 1.0)
            continue
        center = float(np.median(vals))
        robust_scale = float(1.4826 * np.median(np.abs(vals - center)))
        if robust_scale <= 1e-12:
            robust_scale = max(float(np.std(vals)), 1e-12)
        rules[param] = (center - RATIO_MAD_MULTIPLIER * robust_scale,
                        center + RATIO_MAD_MULTIPLIER * robust_scale)
    return rules


def apply_display_rules(combined: pd.DataFrame, rules: dict):
    """Append display_keep and display_exclusion_reason; never remove source rows."""
    df = combined.copy()
    df['display_keep'] = True
    df['display_exclusion_reason'] = ''
    for param in RATIOS:
        mask = df.scope.eq('individual') & df.parameter.eq(param)
        if not mask.any():
            continue
        rows = df.loc[mask]
        truth = numeric(rows.true)
        median = numeric(rows.recovered_median)
        lo, hi = rules[param]
        invalid = ~np.isfinite(truth) | ~np.isfinite(median)
        warning = rows.ratio_warning.fillna('').astype(str).str.len().gt(0).to_numpy()
        extreme = ((truth < lo) | (truth > hi) | (median < lo) | (median > hi))
        reasons = np.where(invalid, 'nonfinite_value',
                   np.where(warning, 'ratio_warning',
                   np.where(extreme, f'extreme_ratio_over_{RATIO_MAD_MULTIPLIER:g}_mad', '')))
        df.loc[mask, 'display_keep'] = (reasons == '')
        df.loc[mask, 'display_exclusion_reason'] = reasons
    return df


def make_display_limits(df: pd.DataFrame):
    """Equal x/y limits across both studies and separate SLURM plotting jobs."""
    limits = {'group': {}, 'individual': {}}
    for scope in ('group', 'individual'):
        for param in PARAMETERS:
            part = df[df.scope.eq(scope) & df.parameter.eq(param)]
            if scope == 'individual':
                if param in RATIOS:
                    part = part[part.display_keep]
                vals = np.r_[numeric(part.true), numeric(part[f'recovered_{preferred_estimator(scope,param)}'])]
            else:
                vals = np.r_[numeric(part.true), numeric(part.recovered_mean),
                             numeric(part.hdi_lower), numeric(part.hdi_upper)]
            limits[scope][param] = padded_limits(vals)
    return limits


def r2_note(ax, r2: float):
    value = f'{r2:.2f}' if np.isfinite(r2) else 'n/a'
    ax.text(0.045, 0.955, rf'$R^2 = {value}$', transform=ax.transAxes,
            ha='left', va='top', fontsize=13, color='#222222',
            bbox=dict(boxstyle='round,pad=0.19', facecolor='white',
                      edgecolor='#d9d9d9', linewidth=0.65, alpha=.91))


def participant_axis(ax, df: pd.DataFrame, row: pd.Series, param: str, limits, color):
    part = df[df.scope.eq('individual') & df.parameter.eq(param)]
    part = part[part.display_keep]
    estimator = preferred_estimator('individual', param)
    x, y = numeric(part.true), numeric(part[f'recovered_{estimator}'])
    good = np.isfinite(x) & np.isfinite(y)
    ax.scatter(x[good], y[good], s=17, alpha=.38, color=color,
               edgecolors='none', rasterized=True)
    ax.plot(limits, limits, ls='--', color='#363636', lw=1.3, alpha=.83)
    ax.set(xlim=limits, ylim=limits)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(LABELS[param], pad=9)
    ax.set_xlabel('generating value')
    ax.set_ylabel('recovered posterior ' + estimator)
    ax.tick_params(axis='both', which='major', labelsize=11.5, length=4)
    r2_note(ax, row.r2)


def group_axis(ax, df: pd.DataFrame, row: pd.Series, param: str, limits, color):
    part = df[df.scope.eq('group') & df.parameter.eq(param)].sort_values('rep')
    rep = numeric(part.rep)
    mu, tr = numeric(part.recovered_mean), numeric(part.true)
    lower, upper = numeric(part.hdi_lower), numeric(part.hdi_upper)
    intervals = np.isfinite(lower) & np.isfinite(upper)
    means = np.isfinite(mu)
    truth_ok = np.isfinite(tr)
    ax.vlines(rep[intervals], lower[intervals], upper[intervals],
              color='#9b9b9b', lw=1.9, alpha=.84)
    ax.scatter(rep[means], mu[means], color=color, s=30,
               label='recovered mean', zorder=4)
    ax.scatter(rep[truth_ok], tr[truth_ok], color='black', marker='x',
               s=32, lw=1.5, label='generating truth', zorder=5)
    ax.set_ylim(limits)
    ax.set_xlim(float(np.min(rep)) - .75, float(np.max(rep)) + .75)
    ax.set_xticks(rep[::max(1, len(rep)//6)].astype(int))
    ax.set_title(LABELS[param], pad=9)
    ax.set_xlabel('recovery replicate')
    ax.set_ylabel('parameter value')
    ax.tick_params(axis='both', which='major', labelsize=11.5, length=4)
    r2_note(ax, row.r2)


def save(fig, out: Path, stem: str, dpi: int):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f'{stem}.pdf', bbox_inches='tight', facecolor='white')
    fig.savefig(out / f'{stem}.png', dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main_overview(df, summary, params, scope, lims, folder, section, study, dpi):
    fig, axes = plt.subplots(2, 4, figsize=(19, 10.7), layout='constrained')
    for ax, param in zip(axes.flat, params):
        row = summary[summary.scope.eq(scope) & summary.parameter.eq(param)].iloc[0]
        if scope == 'individual':
            participant_axis(ax, df, row, param, lims[param], COLORS[study])
        else:
            group_axis(ax, df, row, param, lims[param], COLORS[study])
    for ax in axes.flat[len(params):]:
        ax.axis('off')
    if scope == 'group':
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower right', bbox_to_anchor=(.985,.035),
                   fontsize=12, frameon=False)
    fig.suptitle(f'{study} | {scope}-level recovery | {section}', fontsize=19)
    stem = f'{"participant" if scope == "individual" else "group"}_recovery_{section}'
    save(fig, folder, stem, dpi)


def single_parameter_plots(df, summary, scope, lims, folder, study, dpi):
    for param in PARAMETERS:
        fig, ax = plt.subplots(figsize=(7.1, 6.35), layout='constrained')
        row = summary[summary.scope.eq(scope) & summary.parameter.eq(param)].iloc[0]
        if scope == 'individual':
            participant_axis(ax, df, row, param, lims[param], COLORS[study])
        else:
            group_axis(ax, df, row, param, lims[param], COLORS[study])
            ax.legend(loc='lower right', fontsize=12, frameon=False)
        fig.suptitle(f'{study} | {scope}-level recovery', fontsize=18)
        save(fig, folder, f'{scope}_{param}', dpi)


def write_paper_summary(dest: Path, study: str, summary: pd.DataFrame,
                        detailed: pd.DataFrame, audit: pd.DataFrame,
                        omitted: pd.DataFrame):
    lines = [f'{study}: esaddm + z parameter recovery',
             f'completed recovery replications: {len(audit)}',
             f'fully converged replications (all direct parameters): '
             f'{int(audit.independently_checked_convergence_ok.sum())}/{len(audit)}',
             f'ratio observations excluded from figures only: {len(omitted)}',
             '', 'overview (descriptive summaries of parameter-level recovery):']
    for scope in ('individual', 'group'):
        for section, params in [('fitted', DIRECT),
                                ('derived coefficients', [p for p in DERIVED if p not in RATIOS]),
                                ('derived ratios (median estimates)', [p for p in DERIVED if p in RATIOS])]:
            part = summary[summary.scope.eq(scope) & summary.parameter.isin(params)]
            lines.append(f'  {scope}, {section}: median r={part.r.median():.3f}, '
                         f'r range=[{part.r.min():.3f}, {part.r.max():.3f}], '
                         f'mean r²={part.r2.mean():.3f}, '
                         f'median 95% hdi coverage={part.coverage95.median():.1%}')
    lines.extend(['', 'per-parameter recovery (statistics use all observations, including hidden points):',
                  'scope       parameter                 estimate        r     r²   slope    bias    rmse  cover95  n'])
    for _, row in summary.iterrows():
        lines.append(f'{row.scope:<11} {row.parameter:<25} {row.estimator:<7}'
                     f' {row.r:>7.3f} {row.r2:>6.3f} {row.slope:>7.3f}'
                     f' {row.bias:>7.3f} {row.rmse:>7.3f} {row.coverage95:>8.1%} {int(row.n):>4}')
    lines.extend(['', 'ratio robustness: mean and median recovery statistics',
                  'scope       parameter                 estimate        r     r²    bias     rmse  cover95'])
    for _, row in detailed[detailed.parameter.isin(RATIOS)].iterrows():
        lines.append(f'{row.scope:<11} {row.parameter:<25} {row.estimator:<7}'
                     f' {row.r:>7.3f} {row.r2:>6.3f} {row.bias:>8.3g}'
                     f' {row.rmse:>8.3g} {row.coverage95:>8.1%}')
    lines.extend(['', 'reporting notes:',
                  '- R² is squared Pearson correlation between generating and recovered values;',
                  '  it is not held-out predictive R² and conceals the sign of r.',
                  '- all 7 directly fitted parameters are plotted with posterior means.',
                  '- derived coefficients are plotted with posterior means.',
                  '- the 3 derived ratios are plotted with posterior medians at the individual level;',
                  '  their mean-based metrics remain in recovery_metrics_detailed.csv.',
                  '- group-level plots show posterior means and 95% HDIs.',
                  '- flagged and extremely large ratio points are omitted from FIGURES ONLY;',
                  '  the R² labels, full metrics, coverage, and raw export still include them.',
                  f'- extreme display threshold: {RATIO_MAD_MULTIPLIER:g} x 1.4826 x pooled MAD',
                  '  about the pooled median of true and recovered-median ratio values.',
                  '- omitted points with exact reasons: display_omitted_points.csv.',
                  '- group-level r is based on only the number of recovery replications.',
                  '- participant r pools repeated simulated datasets from the same individuals;',
                  '  observations are not independent participants.',
                  '- in-sample parameter recovery does not prove model discrimination.',
                  '- convergence diagnostics and ratio warnings are not suppressed.'])
    (dest / 'recovery_summary_for_paper.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd(), help='esaddm repository root')
    parser.add_argument('--study', choices=('study1', 'study2', 'both'), default='both')
    parser.add_argument('--out', type=Path, default=None,
                        help='optional output parent; if set uses <out>/study1 and <out>/study2')
    parser.add_argument('--expected-reps', type=int, default=20)
    parser.add_argument('--allow-incomplete', action='store_true', help='diagnostic use only')
    parser.add_argument('--dpi', type=int, default=200)
    args = parser.parse_args()
    if args.expected_reps < 1:
        parser.error('--expected-reps must be positive')
    root = args.root.resolve()
    selected = list(STUDIES if args.study == 'both' else [args.study])

    # Both datasets are loaded to obtain identical per-parameter axes even when
    # the two studies are plotted by independent SLURM batch jobs.
    loaded, audits = {}, {}
    for study in STUDIES:
        loaded[study], audits[study] = load_recovery(root, study,
            args.expected_reps, args.allow_incomplete)
    combined = pd.concat([loaded[s] for s in STUDIES], ignore_index=True)
    combined = apply_display_rules(combined, ratio_display_rules(combined))
    display_limits = make_display_limits(combined)

    for study in selected:
        df = combined[combined.study.eq(study)].copy()
        audit = audits[study]
        dest = ((args.out.resolve() / study) if args.out else
                root / 'studies' / study / 'results' / 'recovery' / MODEL / 'summary')
        dest.mkdir(parents=True, exist_ok=True)

        summary, detailed = make_summaries(df, study)
        hidden = df[df.scope.eq('individual') & ~df.display_keep].copy()
        summary.to_csv(dest / 'recovery_summary.csv', index=False)
        detailed.to_csv(dest / 'recovery_metrics_detailed.csv', index=False)
        audit.to_csv(dest / 'rep_convergence_audit.csv', index=False)
        # This export contains all original observations (no exclusions).
        df.drop(columns=['display_keep', 'display_exclusion_reason']).to_csv(
            dest / 'all_recovery_parameters.csv', index=False)
        hidden.to_csv(dest / 'display_omitted_points.csv', index=False)

        write_paper_summary(dest, study, summary, detailed, audit, hidden)
        figs = dest / 'plots'
        for scope in ('individual', 'group'):
            for section, params in [('fitted', DIRECT), ('derived', DERIVED)]:
                main_overview(df, summary, params, scope, display_limits[scope],
                              figs, section, study, args.dpi)
            # Preserve the established per-parameter output subdirectories.
            subdir = 'participant_individual' if scope == 'individual' else 'group_individual'
            single_parameter_plots(df, summary, scope, display_limits[scope],
                                   figs / subdir, study, args.dpi)

        notes = [f'{study}: esaddm + z recovery',
                 f'completed replications: {len(audit)}',
                 f'convergence failures: {int((~audit.independently_checked_convergence_ok).sum())}',
                 f'ratio warning rows: {int(df.ratio_warning.fillna("").astype(str).str.len().gt(0).sum())}',
                 f'display-only omitted ratio rows: {len(hidden)}',
                 '', 'all 7 fitted parameters + all 7 derived parameters are shown.',
                 'each panel shows R²; see recovery_summary_for_paper.txt for full statistics.',
                 'recovery_summary.csv: primary reported estimator per parameter.',
                 'recovery_metrics_detailed.csv: both mean and median for every parameter.',
                 'display_omitted_points.csv: source rows excluded only from plots.',
                 'rep_convergence_audit.csv: fit convergence by replicate.',
                 'all_recovery_parameters.csv: unfiltered source parameter records.',
                 '', 'ratio panels use posterior medians; original mean values are saved.',
                ]
        (dest / 'README_RESULTS.txt').write_text('\n'.join(notes) + '\n', encoding='utf-8')
        if (~audit.independently_checked_convergence_ok).any():
            warnings.warn(f'{study}: completed replicates with convergence flags; inspect audit')
        if len(hidden):
            warnings.warn(f'{study}: {len(hidden)} ratio observations hidden from figures ONLY; '
                          'see display_omitted_points.csv')
        print(f'completed {study}: {len(audit)} recovery replicates; '
              f'{len(hidden)} plot-only omissions; results: {dest}', flush=True)


if __name__ == '__main__':
    main()
