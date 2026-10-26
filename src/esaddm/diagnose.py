""" diagnostics for fitted ES/EE HDDM models (both studies).

model specifications are read from config/models.json; 
outputs are analysed by default. 
Short smoke fits check if everythign works.
"""
from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import arviz as az
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import expit

from .models import ROOT, load_models

# constants for plots
FILL = '#8FA6B8'
EDGE = '#587286'
REF = '#D95F5F'
TEXT = '#222222'
MAIN_TITLE_SIZE = 28
PANEL_TITLE_SIZE = 22
AXIS_LABEL_SIZE = 22
TICK_LABEL_SIZE = 19

MODEL_LABELS = {
    'accuracy_es': 'Accuracy-coded aDDM (ES)',
    'accuracy_ee': 'Accuracy-coded aDDM (EE)',
    'addm': 'aDDM',
    'addm_z': 'aDDM + z',
    'esaddm': 'ESaDDM',
    'esaddm_z': 'ESaDDM + z',
}
LABELS = {
    'a': ('Boundary separation', r'$a$'),
    't': ('Non-decision time', r'$t$'),
    'z': ('Starting point', 'z'),
    'b_a': ('Attended-value coefficient', r'$b_A$'),
    'b_i': ('Unattended-value coefficient', r'$b_I$'),
    'theta': ('Relative unattended weight', r'$\theta$'),
    'delta_a': ('Attended source difference (S − E)', 'Attended slope difference'),
    'delta_i': ('Unattended source difference (S − E)', 'Unattended slope difference'),
    'b_as': ('Attended S coefficient', r'$b_{AS}$'),
    'b_ae': ('Attended E coefficient', r'$b_{AE}$'),
    'b_is': ('Unattended S coefficient', r'$b_{IS}$'),
    'b_ie': ('Unattended E coefficient', r'$b_{IE}$'),
    'theta_s': ('Relative unattended weight — S', r'$\theta_S$'),
    'theta_e': ('Relative unattended weight — E', r'$\theta_E$'),
    'delta_theta_s_minus_e': ('Relative unattended weight difference (S − E)', r'$\theta_S - \theta_E$'),
}
PLOT_ORDER = ('a', 't', 'z', 'b_a', 'b_i', 'delta_a', 'delta_i',
              'b_as', 'b_ae', 'b_is', 'b_ie', 'theta',
              'theta_s', 'theta_e', 'delta_theta_s_minus_e')


def normalized(name: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', str(name).lower())


def resolve_column(frame: pd.DataFrame, aliases: list[str], *, label: str) -> str:
    """Resolve one group parameter; never substitute a subject/SD coefficient."""
    candidates = [str(c) for c in frame.columns
                  if 'subj' not in str(c).lower()
                  and 'std' not in str(c).lower()
                  and not str(c).startswith('__')]
    for alias in aliases:
        exact = [c for c in candidates if normalized(c) == normalized(alias)]
        if len(exact) == 1:
            return exact[0]
    for alias in aliases:
        prefixed = [c for c in candidates
                    if normalized(c).startswith(normalized(alias))
                    and normalized(c)[len(normalized(alias)):] in ('reg',)]
        if len(prefixed) == 1:
            return prefixed[0]
    raise KeyError(f'Could not uniquely resolve {label}: aliases={aliases}; '
                   f'available group candidates={candidates}')


def extract_fitted_traces(frame: pd.DataFrame, model_name: str, estimate_z: bool) -> dict[str, np.ndarray]:
    """Extract group-level posterior draws without reversing starting-point bias.

    Prefer HDDM's z. If only z_trans exists (we currently don't havea  model that uses z trans but for futuer ref), convert it once
    with the inverse-logit.
    """
    lookup = {
        'a': ['a'],
        't': ['t'],
        'b_a': ['v_attention_w'],
        'b_i': ['v_inattention_w'],
    }
    if model_name.startswith('esaddm'):
        lookup['delta_a'] = ['v_attention_contrast']
        lookup['delta_i'] = ['v_inattention_contrast']
    traces = {}
    for key, aliases in lookup.items():
        col = resolve_column(frame, aliases, label=key)
        traces[key] = np.asarray(pd.to_numeric(frame[col], errors='raise'), dtype=float)

    if estimate_z:
        group_columns = {
            normalized(c): c
            for c in frame.columns
            if "subj" not in str(c).lower()
            and "std" not in str(c).lower()
        }
    
        if "z" in group_columns:
            # HDDM's actual starting-point parameter, already in [0, 1].
            z_col = group_columns["z"]
            z_values = np.asarray(frame[z_col], dtype=float)
    
            if "ztrans" in group_columns:
                z_trans_col = group_columns["ztrans"]
                reconstructed = expit(
                    np.asarray(frame[z_trans_col], dtype=float)
                )
    
                if not np.allclose(
                    z_values, reconstructed, atol=1e-6, rtol=1e-6
                ):
                    raise ValueError(
                        "HDDM z does not agree with expit(z_trans)"
                    )
    
        elif "ztrans" in group_columns:
            # Fallback if only the internal parameter is available.
            z_trans_col = group_columns["ztrans"]
            z_values = expit(
                np.asarray(frame[z_trans_col], dtype=float)
            )
    
        else:
            raise KeyError(
                f"No group-level z or z_trans found for {model_name}"
            )
    
        if not (
            np.isfinite(z_values).all()
            and ((0 <= z_values) & (z_values <= 1)).all()
        ):
            raise ValueError("Starting-point z must lie in [0, 1]")
    
        traces["z"] = z_values
    


    for key, arr in traces.items():
        if not arr.size or not np.isfinite(arr).all():
            raise ValueError(f'{model_name}/{key}: empty or nonfinite sampled values')
    return traces


def derive_source_parameters(base: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Draw-wise algebra for (S−E)/2 contrast-coded source regressors."""
    a, i = base['b_a'], base['b_i']
    if len(a) != len(i):
        raise ValueError('Unequal attended/unattended trace lengths')
    if 'delta_a' not in base:
        with np.errstate(divide='ignore', invalid='ignore'):
            return {'theta': np.divide(i, a, out=np.full_like(i, np.nan), where=np.abs(a) > 1e-10)}
    da, di = base['delta_a'], base['delta_i']
    if not (len(a) == len(da) == len(di)):
        raise ValueError('Source contrasts and attended/unattended draws are not aligned')
    b_as = a + da / 2.0
    b_ae = a - da / 2.0
    b_is = i + di / 2.0
    b_ie = i - di / 2.0
    with np.errstate(divide='ignore', invalid='ignore'):
        theta_s = np.divide(b_is, b_as, out=np.full_like(b_is, np.nan), where=np.abs(b_as) > 1e-10)
        theta_e = np.divide(b_ie, b_ae, out=np.full_like(b_ie, np.nan), where=np.abs(b_ae) > 1e-10)
    return {
        'b_as': b_as, 'b_ae': b_ae, 'b_is': b_is, 'b_ie': b_ie,
        'theta_s': theta_s, 'theta_e': theta_e,
        'delta_theta_s_minus_e': theta_s - theta_e,
    }


def reference_value(key: str) -> float | None:
    """Null/neutral reference lines: NOT zero crossings of HDIs."""
    if key == 'z':
        return 0.5
    if key in ('theta', 'theta_s', 'theta_e'):
        return 1.0  # no attentional discounting
    if key in ('a', 't'):
        return None  # zero is not an informative reference for these positive parameters
    return 0.0


def hdi_interval(x: np.ndarray, mass: float = 0.95) -> tuple[float, float]:
    """Short-window HDI matching the original presentation plot implementation."""
    vals = np.sort(np.asarray(x, dtype=float)[np.isfinite(x)])
    if not len(vals):
        raise ValueError('No finite posterior draws')
    window = int(np.floor(mass * len(vals)))
    if window < 1 or window >= len(vals):
        return tuple(map(float, np.quantile(vals, [(1-mass)/2, (1+mass)/2])))
    widths = vals[window:] - vals[:-window]
    start = int(np.argmin(widths))
    return float(vals[start]), float(vals[start + window])


def posterior_stats(per_chain: list[np.ndarray], key: str) -> dict:
    finite = [np.asarray(v, dtype=float)[np.isfinite(v)] for v in per_chain]
    if any(len(v) < 4 for v in finite):
        raise ValueError(f'{key}: not enough valid draws for all chains')
    draws = np.concatenate(finite)
    hdi = hdi_interval(draws, mass=0.95)
    reference = reference_value(key)
    prob_above = float((draws > reference).mean()) if reference is not None else np.nan
    prob_below = float((draws < reference).mean()) if reference is not None else np.nan
    p_zero_above = float((draws > 0).mean())
    p_zero_below = float((draws < 0).mean())
    stats = {
        'parameter': key,
        'mean': float(draws.mean()),
        'median': float(np.median(draws)),
        'sd': float(draws.std(ddof=1)),
        'hdi_lower': float(hdi[0]),
        'hdi_upper': float(hdi[-1]),
        'p_gt_0': p_zero_above,
        'p_lt_0': p_zero_below,
        'p_direction_vs_zero': max(p_zero_above, p_zero_below),
        # p_direction uses the scientifically meaningful reference when one
        # exists: e.g. z=0.5, theta=1, and source contrasts=0.
        'p_direction': max(prob_above, prob_below) if reference is not None else np.nan,
        'n_valid_draws': int(len(draws)),
        'reference': reference,
        'p_gt_reference': prob_above,
        'p_lt_reference': prob_below,
        'p_direction_reference': max(prob_above, prob_below) if reference is not None else np.nan,
        'hdi_contains_reference': bool(hdi[0] <= reference <= hdi[1]) if reference is not None else None,
        'r_hat': np.nan,
        'ess_bulk': np.nan,
        'ess_tail': np.nan,
    }
    if len(finite) >= 2:
        n = min(map(len, finite))
        array = np.stack([v[:n] for v in finite])
        # R-hat/ESS calculated from individual chains; never from concatenated samples.
        if n >= 8:
            stats.update({
                'r_hat': float(np.asarray(az.rhat(array))),
                'ess_bulk': float(np.asarray(az.ess(array, method='bulk'))),
                'ess_tail': float(np.asarray(az.ess(array, method='tail', prob=(.05, .95)))),
            })
    return stats


def apply_talk_style(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.tick_params(axis='both', labelsize=TICK_LABEL_SIZE, width=1.1)
    # Exactly the original talk-ready orientation: parameter along x, density along y.
    ax.set_ylabel('Posterior density', fontsize=AXIS_LABEL_SIZE, color=TEXT)
    ax.grid(False)


def plot_panel(ax, key: str, draws: np.ndarray):
    clean = np.asarray(draws, dtype=float)
    clean = clean[np.isfinite(clean)]
    row = posterior_stats([clean], key)
    title, xlabel = LABELS[key]
    ax.hist(clean, bins=45, density=True, color=FILL, edgecolor='none', alpha=.82)
    ax.axvline(row['mean'], color=EDGE, linewidth=2.2)
    reference = reference_value(key)
    if reference is not None:
        ax.axvline(reference, color=REF, linestyle=':', linewidth=2.6, alpha=.72)
    ax.set_title(f"{title}\nmean = {row['mean']:.3f}, "
                 f"95% HDI [{row['hdi_lower']:.3f}, {row['hdi_upper']:.3f}]",
                 fontsize=PANEL_TITLE_SIZE, color=TEXT, pad=12)
    ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_SIZE, color=TEXT)
    apply_talk_style(ax)


def save_figure(fig, path_without_suffix: Path) -> None:
    fig.savefig(path_without_suffix.with_suffix('.png'), dpi=300,
                bbox_inches='tight', facecolor='white')
    fig.savefig(path_without_suffix.with_suffix('.pdf'),
                bbox_inches='tight', facecolor='white')
    plt.close(fig)


def custom_posterior_plots(study: str, model: str, all_traces: dict[str, list[np.ndarray]],
                           out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    heading = f'{MODEL_LABELS[model]} — {study.replace("study", "Study ")}'
    available = [key for key in PLOT_ORDER if key in all_traces]
    for key in available:
        fig, ax = plt.subplots(figsize=(9.2, 6.2))
        plot_panel(ax, key, np.concatenate(all_traces[key]))
        fig.suptitle(heading, fontsize=MAIN_TITLE_SIZE, y=1.03)
        fig.tight_layout()
        save_figure(fig, out_dir / f'{model}_{study}_custom_{key}')

    # The two horizontal three-panel layouts reproduce the original talk-ready script.
    if model == 'esaddm_z':
        groups = [('main', ('delta_a', 'delta_i', 'z')),
                  ('theta', ('theta_e', 'theta_s', 'delta_theta_s_minus_e'))]
    elif model == 'esaddm':
        groups = [('main', ('delta_a', 'delta_i', 'delta_theta_s_minus_e')),
                  ('theta', ('theta_e', 'theta_s'))]
    else:
        groups = [('main', tuple(k for k in ('b_a', 'b_i', 'z', 'theta') if k in all_traces))]
    for group_name, keys in groups:
        if not keys:
            continue
        fig, axes = plt.subplots(1, len(keys), figsize=(8 * len(keys), 6.3), squeeze=False)
        for ax, key in zip(axes[0], keys):
            plot_panel(ax, key, np.concatenate(all_traces[key]))
        fig.suptitle(heading, fontsize=MAIN_TITLE_SIZE, y=1.03)
        fig.tight_layout()
        save_figure(fig, out_dir / f'{model}_{study}_custom_{group_name}_horizontal')


def plot_trace_overview(study: str, model: str,
                        all_traces: dict[str, list[np.ndarray]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = [k for k in PLOT_ORDER if k in all_traces]
    if not keys:
        return
    ncols = 2
    nrows = (len(keys) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.7 * nrows), squeeze=False)
    for ax, key in zip(axes.flat, keys):
        for ci, values in enumerate(all_traces[key], start=1):
            ax.plot(values, linewidth=.65, alpha=.75, label=f'Chain {ci}')
        ax.set_title(LABELS[key][0], fontsize=15)
        ax.set_xlabel('Posterior draw', fontsize=12)
        ax.tick_params(labelsize=11)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    for ax in list(axes.flat)[len(keys):]:
        ax.axis('off')
    axes.flat[0].legend(fontsize=10, ncol=min(5, len(all_traces[keys[0]])))
    fig.suptitle(f'{MODEL_LABELS[model]} — {study} — chain traces', fontsize=20)
    fig.tight_layout()
    save_figure(fig, out_dir / f'{model}_{study}_chain_trace_overview')


def diagnose_model(study: str, model_name: str, *, chains: int, stage: str) -> None:
    import hddm
    import kabuki

    spec = load_models()[model_name]
    if stage not in ('fits', 'smoke'):
        raise ValueError('stage must be fits or smoke')
    result_root = ROOT / 'studies' / study / 'results'
    model_dir = result_root / stage / model_name
    prefix = f'{model_name}_{study}' + ('_smoke' if stage == 'smoke' else '')
    chain_paths = [model_dir / f'{prefix}_chain{i}.hddm' for i in range(1, chains + 1)]
    manifest = model_dir / f'{prefix}_run.json'

    missing = [str(p) for p in chain_paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f'Missing fitted chains for {study}/{model_name}: {missing}')
    if not manifest.is_file():
        raise FileNotFoundError(f'Missing completed fit manifest: {manifest}')
    run_meta = json.loads(manifest.read_text(encoding='utf-8'))
    expected_chains = list(range(1, chains + 1))
    if run_meta.get('status') != 'completed' or run_meta.get('chains') != expected_chains:
        raise ValueError(f'Fit has not completed the expected {chains} chains: {manifest}')
    # Reject stale/mismatched results, especially if models.json changed
    # after fitting (e.g. fixing z in the accuracy-coded models).
    for field, expected in (
        ('study', study),
        ('model', model_name),
        ('coordinate', spec.coordinate),
        ('formula', spec.drift_formula),
        ('included_ddm_parameters', spec.include),
    ):
        if run_meta.get(field) != expected:
            raise ValueError(
                f'Fit manifest mismatch for {field}: '
                f'saved={run_meta.get(field)!r}, current={expected!r}. '
                'Do not diagnose models using a different model specification.'
            )
    if stage == 'smoke':
        print('WARNING: diagnosing smoke data, not scientifically valid posteriors.', flush=True)

    output = result_root / 'diagnostics' / model_name / stage
    output.mkdir(parents=True, exist_ok=True)
    print(f'LOADING {study}/{model_name}: {chains} chains from {model_dir}', flush=True)
    models = [hddm.load(str(p)) for p in chain_paths]
    per_chain = {}
    for i, obj in enumerate(models, start=1):
        frame = obj.get_traces()
        draws = extract_fitted_traces(frame, model_name, spec.estimate_z)
        draws.update(derive_source_parameters(draws))
        for key, array in draws.items():
            per_chain.setdefault(key, []).append(array)
        print(f'  Loaded chain {i}: {len(frame)} retained draws', flush=True)

    if chains >= 2:
        all_rhat = hddm.analyze.gelman_rubin(models)
        pd.Series(all_rhat, name='r_hat').to_csv(output / f'{prefix}_gelman_rubin_all_nodes.csv')

    # Use the exact native diagnostics from the existing HDDM workflow.
    combined = kabuki.utils.concat_models(models)
    combined.gen_stats().to_csv(output / f'{prefix}_posterior_summary_all_nodes.csv')
    try:
        dic_value = float(combined.dic)
        (output / f'{prefix}_dic.txt').write_text(f'{dic_value:.12f}\n', encoding='utf-8')
    except Exception as exc:
        dic_value = np.nan
        (output / f'{prefix}_dic.txt').write_text(f'unavailable: {exc}\n', encoding='utf-8')

    rows = [posterior_stats(per_chain[k], k) for k in PLOT_ORDER if k in per_chain]
    summary = pd.DataFrame(rows)
    summary['dic'] = dic_value
    # Ratios of posterior coefficients can become unstable when a denominator
    # approaches zero. Keep the diagnostic WITH the ordinary CSV summary instead
    # of creating a separate warning JSON file. The 0.1 threshold is only an
    # interpretable scale-dependent diagnostic, not a formal validity cut-off.
    denominator_keys = ['b_as', 'b_ae'] if 'delta_a' in per_chain else ['b_a']
    summary['p_abs_denominator_below_0p1'] = np.nan
    for key in denominator_keys:
        draws = np.concatenate(per_chain[key])
        p_small = float(np.mean(np.abs(draws) < .1))
        summary.loc[summary['parameter'] == key, 'p_abs_denominator_below_0p1'] = p_small
        if p_small > .01:
            print(f'CAUTION {study}/{model_name}: {key} has '
                  f'{p_small:.1%} of draws with abs(denominator) < 0.1; '
                  'derived theta ratios may be unstable.', flush=True)
    summary.to_csv(output / f'{prefix}_core_posterior_summary.csv',
                   index=False, float_format='%.8f')

    # Keep the original HDDM posterior plots, but NOT the duplicate old custom posterior_plots/.
    hd = output / 'hddm_posterior_plots'
    hd.mkdir(parents=True, exist_ok=True)
    combined.plot_posteriors(save=True, path=str(hd), format='pdf')
    custom_posterior_plots(study, model_name, per_chain, output / 'custom')
    traces = output / 'traces'
    traces.mkdir(parents=True, exist_ok=True)
    plot_trace_overview(study, model_name, per_chain, traces)

    print(f'PASS diagnostics {study}/{model_name}: {output}', flush=True)
    del combined, models
    gc.collect()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description='HDDM model diagnostics and original talk-ready posterior plots (no PPC)')
    parser.add_argument('--study', required=True, choices=('study1', 'study2'))
    parser.add_argument('--model', required=True, choices=(*load_models(), 'all'))
    parser.add_argument('--chains', type=int, default=5)
    parser.add_argument('--stage', choices=('fits', 'smoke'), default='fits')
    args = parser.parse_args(argv)
    if args.chains < 1:
        parser.error('Positive chains are required')
    names = list(load_models()) if args.model == 'all' else [args.model]
    for name in names:
        diagnose_model(args.study, name, chains=args.chains, stage=args.stage)
    print(f'PASS: diagnostics completed for {len(names)} models in {args.study}', flush=True)


if __name__ == '__main__':
    main()
