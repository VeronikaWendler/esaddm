"""Tests of posterior transformations"""
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from esaddm.diagnose import derive_source_parameters, extract_fitted_traces, reference_value, posterior_stats


def test_derived_source_slopes_and_theta_are_draw_aligned():
    base = {
        'b_a': np.array([4.0, 4.0]), 'b_i': np.array([2.0, 2.0]),
        'delta_a': np.array([2.0, -2.0]), 'delta_i': np.array([2.0, 2.0]),
    }
    out = derive_source_parameters(base)
    np.testing.assert_allclose(out['b_as'], [5., 3.])
    np.testing.assert_allclose(out['b_ae'], [3., 5.])
    np.testing.assert_allclose(out['b_is'], [3., 3.])
    np.testing.assert_allclose(out['b_ie'], [1., 1.])
    np.testing.assert_allclose(out['theta_s'], [3/5, 1.])
    np.testing.assert_allclose(out['theta_e'], [1/3, 1/5])


def test_z_is_from_group_z_trans_and_not_hierarchical_sd():
    d = pd.DataFrame({'a': [2., 2.], 't': [.3, .3],
                      'v_attention_w': [3., 3.], 'v_inattention_w': [1., 1.],
                      'z_trans': [0., 1.], 'z_std': [.4, .5]})
    out = extract_fitted_traces(d, 'addm_z', True)
    np.testing.assert_allclose(out['z'], expit([0., 1.]))


def test_reference_lines():
    assert reference_value('delta_i') == 0
    assert reference_value('delta_theta_s_minus_e') == 0
    assert reference_value('z') == .5
    assert reference_value('theta_e') == 1
    assert reference_value('a') is None


def test_posterior_stats_from_separate_chains():
    rng = np.random.default_rng(42)
    row = posterior_stats([rng.normal(size=300), rng.normal(size=300)], 'b_a')
    assert 0 < row['ess_bulk'] <= 600
    assert .9 < row['r_hat'] < 1.2
    assert row['hdi_lower'] < row['hdi_upper']


def test_custom_plot_matches_original_talk_ready_axes_and_reference():
    import matplotlib.pyplot as plt
    from esaddm.diagnose import plot_panel, REF
    fig, ax = plt.subplots()
    plot_panel(ax, 'delta_a', np.random.default_rng(11).normal(size=1000))
    assert ax.get_ylabel() == 'Posterior density'
    assert ax.get_xlabel() == 'Attended slope difference'
    assert ax.patches  # vertical density histogram
    ref_lines = [line for line in ax.lines if line.get_color() == REF]
    assert len(ref_lines) == 1
    assert ref_lines[0].get_linestyle() == ':'
    assert np.allclose(ref_lines[0].get_xdata(), [0, 0])
    plt.close(fig)


def test_diagnostics_stays_independent_of_ppc():
    from pathlib import Path
    import esaddm.diagnose as diagnose
    source = Path(diagnose.__file__).read_text()
    assert 'plot_posterior_predictive(' not in source
    assert "--ppc-samples" not in source
    assert "--skip-ppc" not in source
