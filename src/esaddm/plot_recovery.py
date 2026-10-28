#!/usr/bin/env python3
"""Plot ALL 14 ESaDDM + z recovery parameters, separately for each study.

Reads existing recovery_parameters.csv and complete.json from rep_00 .. rep_09.
No HDDM, no synthetic data regeneration and no model fitting.

Default outputs:
  studies/study1/results/recovery/esaddm_z/summary/
  studies/study2/results/recovery/esaddm_z/summary/

Axes for the same parameter are identical across studies.
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
PARAMETERS = DIRECT + DERIVED
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
    'recovered_mean', 'hdi_lower', 'hdi_upper', 'r_hat',
    'ess_bulk', 'ess_tail', 'covered_95',
}
STUDIES = ('study1', 'study2')
EXPECTED_SUBJECTS = {'study1': 21, 'study2': 20}


def load_recovery(root: Path, study: str, reps: int, allow_incomplete: bool):
    frames, audits = [], []
    parent = root / 'studies' / study / 'results' / 'recovery' / 'esaddm_z'
    for rep in range(reps):
        folder = parent / f'rep_{rep:02d}'
        csv, manifest = folder / 'recovery_parameters.csv', folder / 'complete.json'
        if not csv.is_file() or not manifest.is_file():
            if allow_incomplete:
                warnings.warn(f'missing replicate, not plotted: {folder}')
                continue
            raise FileNotFoundError(f'missing recovery_parameters.csv or complete.json: {folder}')
        meta = json.loads(manifest.read_text())
        if meta.get('status') != 'completed':
            if allow_incomplete:
                warnings.warn(f'incomplete replicate, not plotted: {folder}')
                continue
            raise ValueError(f'replicate not marked completed: {manifest}')
        df = pd.read_csv(csv)
        missing = REQUIRED - set(df.columns)
        if missing:
            raise ValueError(f'{csv}: missing columns {sorted(missing)}')
        if df.study.nunique() != 1 or df.study.iloc[0] != study:
            raise ValueError(f'{csv}: study mismatch')
        if df.rep.nunique() != 1 or int(df.rep.iloc[0]) != rep:
            raise ValueError(f'{csv}: replicate mismatch')
        if df.duplicated(['scope','subj_idx','parameter']).any():
            raise ValueError(f'{csv}: duplicate parameter records')
        if set(df.parameter.unique()) != set(PARAMETERS):
            raise ValueError(f'{csv}: parameter set does not match all 14 parameters')
        for scope, n in [('group', 1), ('individual', EXPECTED_SUBJECTS[study])]:
            for p in PARAMETERS:
                sub = df[df.scope.eq(scope) & df.parameter.eq(p)]
                if len(sub) != n:
                    raise ValueError(f'{csv}: expected {n} rows for {scope}/{p}, got {len(sub)}')
        df['study'] = study
        df['rep'] = rep
        if 'ratio_warning' not in df:
            df['ratio_warning'] = ''
        if 'denominator_ci95_contains_zero' not in df:
            df['denominator_ci95_contains_zero'] = np.nan
        if 'fraction_finite_posterior_draws' not in df:
            df['fraction_finite_posterior_draws'] = np.nan
        d = df[df.parameter.isin(DIRECT)]
        converged_calculated = bool((pd.to_numeric(d.r_hat,errors='coerce').le(1.01)
            & pd.to_numeric(d.ess_bulk,errors='coerce').ge(400)
            & pd.to_numeric(d.ess_tail,errors='coerce').ge(400)).all())
        audits.append(dict(study=study, rep=rep, n_rows=len(df),
            manifest_convergence_ok=meta.get('direct_convergence_ok'),
            independently_checked_convergence_ok=converged_calculated,
            max_direct_rhat=pd.to_numeric(d.r_hat, errors='coerce').max(),
            min_direct_ess_bulk=pd.to_numeric(d.ess_bulk, errors='coerce').min(),
            min_direct_ess_tail=pd.to_numeric(d.ess_tail, errors='coerce').min(),
            n_ratio_warning_rows=int(df.ratio_warning.fillna('').astype(str).str.len().gt(0).sum())))
        frames.append(df)
    if not frames:
        raise ValueError(f'no completed recovery replicates found for {study} in {parent}')
    return pd.concat(frames, ignore_index=True), pd.DataFrame(audits)


def _coverage(s):
    vals = s.replace({'true': 1, 'false': 0, 'True': 1, 'False': 0})
    return pd.to_numeric(vals, errors='coerce')


def statistics(df):
    records = []
    for (scope, param), x in df.groupby(['scope', 'parameter'], sort=False):
        v = x[['true', 'recovered_mean']].apply(pd.to_numeric, errors='coerce')
        ok = np.isfinite(v.true.to_numpy()) & np.isfinite(v.recovered_mean.to_numpy())
        tr = v.true.to_numpy()[ok]
        recovered = v.recovered_mean.to_numpy()[ok]
        n = len(tr)
        r = float(np.corrcoef(tr, recovered)[0,1]) if n >= 3 and np.std(tr)>0 and np.std(recovered)>0 else np.nan
        slope = float(np.polyfit(tr, recovered, 1)[0]) if n >= 2 and np.std(tr)>0 else np.nan
        within_rep_r = []
        for _, g in x.groupby('rep'):
            t = pd.to_numeric(g.true, errors='coerce').to_numpy()
            y = pd.to_numeric(g.recovered_mean, errors='coerce').to_numpy()
            good = np.isfinite(t) & np.isfinite(y)
            if good.sum() >= 3 and np.std(t[good]) > 0 and np.std(y[good]) > 0:
                within_rep_r.append(np.corrcoef(t[good], y[good])[0,1])
        cov = _coverage(x.covered_95)
        records.append(dict(scope=scope, parameter=param, n=n,
            n_reps=int(x.rep.nunique()), r=r,
            median_within_rep_r=float(np.median(within_rep_r)) if within_rep_r else np.nan,
            slope=slope,
            bias=float(np.mean(recovered-tr)) if n else np.nan,
            rmse=float(np.sqrt(np.mean((recovered-tr)**2))) if n else np.nan,
            mae=float(np.mean(np.abs(recovered-tr))) if n else np.nan,
            coverage95=float(cov.mean()), coverage_n=int(cov.notna().sum()),
            median_rhat=float(pd.to_numeric(x.r_hat, errors='coerce').median()),
            max_rhat=float(pd.to_numeric(x.r_hat, errors='coerce').max()),
            min_bulk_ess=float(pd.to_numeric(x.ess_bulk, errors='coerce').min()),
            min_tail_ess=float(pd.to_numeric(x.ess_tail, errors='coerce').min()),
            n_ratio_warnings=int(x.ratio_warning.fillna('').astype(str).str.len().gt(0).sum())))
    out = pd.DataFrame(records)
    out['parameter'] = pd.Categorical(out.parameter, categories=PARAMETERS, ordered=True)
    return out.sort_values(['scope', 'parameter']).reset_index(drop=True)


def axis_limits(all_data, scope):
    result = {}
    d = all_data[all_data.scope.eq(scope)]
    for param in PARAMETERS:
        x = d[d.parameter.eq(param)]
        cols = ['true','recovered_mean'] + (['hdi_lower','hdi_upper'] if scope == 'group' else [])
        z = np.concatenate([pd.to_numeric(x[c], errors='coerce').to_numpy() for c in cols])
        z = z[np.isfinite(z)]
        if len(z) == 0:
            result[param] = (-1, 1)
            continue
        lo, hi = float(np.min(z)), float(np.max(z))
        pad = max((hi-lo)*.065, .012*max(abs(lo),abs(hi),1))
        result[param] = (lo-pad, hi+pad)
    return result


def label_stats(row):
    if row is None:
        return 'no recovery summary'
    return (f"r = {row.r:.2f}   slope = {row.slope:.2f}\n"
            f"bias = {row.bias:.3g}   rmse = {row.rmse:.3g}\n"
            f"95% hdi coverage = {row.coverage95:.0%} (n={row.coverage_n})")


def stats_row(summary, scope, param):
    x = summary[summary.scope.eq(scope) & summary.parameter.astype(str).eq(param)]
    return x.iloc[0] if len(x) else None


def participant_axis(ax, df, summary, param, lim, color):
    x = df[(df.scope == 'individual') & df.parameter.eq(param)]
    tr = pd.to_numeric(x.true, errors='coerce').to_numpy()
    y = pd.to_numeric(x.recovered_mean, errors='coerce').to_numpy()
    good = np.isfinite(tr) & np.isfinite(y)
    ax.scatter(tr[good], y[good], s=12, alpha=.36, color=color, edgecolors='none', rasterized=True)
    ax.plot(lim, lim, '--', color='black', linewidth=1.1, alpha=.8)
    ax.set(xlim=lim, ylim=lim)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(LABELS[param], fontsize=13)
    ax.tick_params(labelsize=8)
    ax.text(.03,.97, label_stats(stats_row(summary,'individual',param)),
            transform=ax.transAxes, va='top', ha='left', fontsize=8,
            bbox=dict(facecolor='white', edgecolor='0.84', alpha=.90,
                      boxstyle='round,pad=.3'))
    if stats_row(summary,'individual',param).n_ratio_warnings:
        ax.text(.98,.02, 'ratio warning(s): see csv', ha='right', va='bottom',
                transform=ax.transAxes, fontsize=7, color='firebrick')


def group_axis(ax, df, summary, param, lim, color):
    x = df[(df.scope == 'group') & df.parameter.eq(param)].sort_values('rep')
    rep = x.rep.to_numpy(int)
    mu = pd.to_numeric(x.recovered_mean, errors='coerce').to_numpy()
    tr = pd.to_numeric(x.true, errors='coerce').to_numpy()
    lower = pd.to_numeric(x.hdi_lower, errors='coerce').to_numpy()
    upper = pd.to_numeric(x.hdi_upper, errors='coerce').to_numpy()
    good = np.isfinite(mu) & np.isfinite(lower) & np.isfinite(upper)
    ax.vlines(rep[good], lower[good], upper[good], color='.52', alpha=.78, lw=2.1)
    ax.scatter(rep[good], mu[good], color=color, s=26, zorder=4, label='recovered mean')
    ax.scatter(rep, tr, color='black', marker='x', s=25, lw=1.5, zorder=5, label='generating truth')
    ax.set_ylim(lim)
    ax.set_xlim(min(rep)-.65, max(rep)+.65)
    ax.set_xticks(rep[::max(1,len(rep)//6)])
    ax.tick_params(labelsize=8)
    ax.set_title(LABELS[param], fontsize=13)
    ax.text(.03,.97, label_stats(stats_row(summary,'group',param)),
            transform=ax.transAxes, va='top', ha='left', fontsize=8,
            bbox=dict(facecolor='white',edgecolor='0.84',alpha=.90,
                      boxstyle='round,pad=.3'))


def save(fig, out, stem, dpi):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f'{stem}.pdf', bbox_inches='tight', facecolor='white')
    fig.savefig(out / f'{stem}.png', dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def overview(df, summ, params, scope, limits, out, name, study, color, dpi):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.5), layout='constrained')
    for ax, param in zip(axes.flat, params):
        if scope == 'individual':
            participant_axis(ax, df, summ, param, limits[param], color)
            ax.set_xlabel('generating value', fontsize=10)
            ax.set_ylabel('recovered posterior mean', fontsize=10)
        else:
            group_axis(ax, df, summ, param, limits[param], color)
            ax.set_xlabel('recovery replicate', fontsize=10)
            ax.set_ylabel('parameter value', fontsize=10)
    for ax in axes.flat[len(params):]:
        ax.axis('off')
    if scope == 'group':
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower right', bbox_to_anchor=(.98,.05), frameon=False, fontsize=10)
    fig.suptitle(f'{study} | {scope}-level recovery | {name}', fontsize=15)
    save(fig, out, f'{"participant" if scope == "individual" else "group"}_recovery_{name}', dpi)


def one_parameter_figures(df, summ, scope, lims, out, study, color, dpi):
    out.mkdir(parents=True, exist_ok=True)
    for param in PARAMETERS:
        fig, ax = plt.subplots(figsize=(5.4, 4.9), layout='constrained')
        if scope == 'individual':
            participant_axis(ax, df, summ, param, lims[param], color)
            ax.set_xlabel('generating value')
            ax.set_ylabel('recovered posterior mean')
        else:
            group_axis(ax, df, summ, param, lims[param], color)
            ax.set_xlabel('recovery replicate')
            ax.set_ylabel('parameter value')
            ax.legend(loc='lower right', fontsize=8, frameon=False)
        fig.suptitle(f'{study} | {LABELS[param]} | {scope} recovery', fontsize=14)
        save(fig, out, f'{scope}_{param}', dpi)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd(), help='esaddm repository root')
    parser.add_argument('--study', choices=('study1','study2','both'), default='both')
    parser.add_argument('--out', type=Path, default=None,
        help='optional output parent; if set, uses <out>/study1 and <out>/study2')
    parser.add_argument('--expected-reps', type=int, default=10)
    parser.add_argument('--allow-incomplete', action='store_true', help='diagnostic use only')
    parser.add_argument('--dpi', type=int, default=190)
    args = parser.parse_args()
    root = args.root.resolve()
    selected = list(STUDIES if args.study == 'both' else [args.study])

    # Both study datasets are loaded for common parameter axes even when
    # their plots are generated in separate SLURM jobs.
    loaded, audits = {}, {}
    for study in STUDIES:
        loaded[study], audits[study] = load_recovery(root, study, args.expected_reps, args.allow_incomplete)

    combined = pd.concat([loaded[s] for s in STUDIES], ignore_index=True)
    participant_lims = axis_limits(combined,'individual')
    group_lims = axis_limits(combined,'group')
    for study in selected:
        df, audit = loaded[study], audits[study]
        dest = ((args.out.resolve() / study) if args.out else
                root / 'studies' / study / 'results' / 'recovery' / 'esaddm_z' / 'summary')
        dest.mkdir(parents=True, exist_ok=True)
        summ = statistics(df)
        summ.insert(0, 'study', study)
        summ.to_csv(dest/'recovery_summary.csv', index=False)
        audit.to_csv(dest/'rep_convergence_audit.csv', index=False)
        df.to_csv(dest/'all_recovery_parameters.csv', index=False)
        color = '#3977A9' if study=='study1' else '#B96057'
        figs = dest / 'plots'
        for scope, limits in [('individual',participant_lims),('group',group_lims)]:
            for params, label in [(DIRECT,'fitted'),(DERIVED,'derived')]:
                overview(df,summ,params,scope,limits,figs,label,study,color,args.dpi)
            subdir = 'participant_individual' if scope=='individual' else 'group_individual'
            one_parameter_figures(df,summ,scope,limits,figs/subdir,study,color,args.dpi)
        any_conv_failed = bool((~audit.independently_checked_convergence_ok).any())
        flagged = int(df.ratio_warning.fillna('').astype(str).str.len().gt(0).sum())
        note = [f'{study}: esaddm + z recovery',f'completed replications: {len(audit)}',
                f'convergence failures: {int((~audit.independently_checked_convergence_ok).sum())}',
                f'ratio warning rows: {flagged}',
                '', 'plots contain all 7 directly fitted + all 7 derived parameters.',
                'overview: plots/participant_recovery_fitted.*, plots/participant_recovery_derived.*',
                '          plots/group_recovery_fitted.*, plots/group_recovery_derived.*',
                'individual: plots/participant_individual/ and plots/group_individual/',
                '', 'each parameter uses the same axis limits across study1 and study2 when --study both.',
                'r and slope pool replicate x participant observations (descriptive only).',
                'each participant is repeated across recovery replications; these are not independent subjects.',
                'group-level interval coverage is based on 10 replicates and is imprecise.',
                'in-sample recovery cannot establish cross-model discrimination.',
                'convergence failures and unstable theta ratios are retained and flagged, not hidden.']
        (dest/'README_RESULTS.txt').write_text('\n'.join(note)+'\n', encoding='utf-8')
        if any_conv_failed:
            warnings.warn(f'{study}: some completed replicates failed convergence; inspect audit')
        if flagged:
            warnings.warn(f'{study}: ratio warnings in {flagged} parameter rows; inspect source table')
        print(f'completed {study}: {len(audit)} recovery replications; 7 fitted + 7 derived parameters; figures/tables: {dest}', flush=True)


if __name__ == '__main__':
    main()
