"""PPC transformation/alignment checks; real HDDM sampling is BlueBEAR-only."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from esaddm.models import ROOT, check_input, load_models
from esaddm.ppc import (
    OBS_E, OBS_S, PPC_E, PPC_S, TRIAL_KEYS, attach_predictive_outcomes,
    bin_means, observed_metrics, prepare_observed, rank_quintile,
    split_draws, summarise_metrics, subject_boxplot_data, draw_metrics,
)


def test_colors_and_replicate_allocation():
    assert (OBS_E, PPC_E) == ('deepskyblue', 'steelblue')
    assert (OBS_S, PPC_S) == ('darkorchid', 'indigo')
    assert split_draws(250, 5) == [50, 50, 50, 50, 50]
    assert split_draws(253, 5) == [51, 51, 51, 50, 50]


def test_quintiles_stable_for_equal_values():
    groups = rank_quintile(pd.Series([1.0] * 26))
    assert sorted(pd.Series(groups).value_counts().tolist()) == [5, 5, 5, 5, 6]
    assert set(groups) == {1, 2, 3, 4, 5}


@pytest.mark.parametrize('study', ['study1', 'study2'])
@pytest.mark.parametrize('model', ['accuracy_es', 'accuracy_ee', 'esaddm_z'])
def test_real_prepared_trial_metadata(study, model):
    spec = load_models()[model]
    if not (ROOT / 'studies' / study / 'data' / 'prepared' / spec.input_name).exists():
        pytest.skip('Research datasets are not part of public code repository')
    prepared, _ = check_input(study, spec)
    obs = prepare_observed(study, model, prepared, prepared)
    assert len(obs) == len(prepared)
    rows, subjects = observed_metrics(obs, study, model)
    assert rows and subjects
    if model == 'esaddm_z':
        assert obs.is_tie.any()
        assert obs.accuracy_value.isna().sum() == int(obs.is_tie.sum())
        assert obs.accuracy_value.notna().sum() + obs.is_tie.sum() == len(obs)
        assert np.allclose(obs.dwell_ms_s_minus_e,
                           obs.Dwell_S-obs.Dwell_E, rtol=0, atol=1e-7)
    if model == 'accuracy_ee':
        assert set(obs.choice_group) == {'Correct', 'Error'}
    if model == 'accuracy_es':
        assert set(obs.choice_group) == {'E', 'S'}


def test_predictive_alignment_is_keyed_not_positional():
    obs = pd.DataFrame({
        'sub_id': [1]*5 + [2]*5,
        'phase': ['ES']*10, 'trial': list(range(1,6))*2,
        'subj_idx': [1]*5+[2]*5,
        'rt': np.arange(1,11, dtype=float),
        'response': [0,1]*5,
        'better_source': ['S']*10,
        'worse_source': ['E']*10,
        'is_tie': [False]*10,
        'dwell_quintile': [1,2,3,4,5]*2,
    })
    predictions = obs[[*TRIAL_KEYS,'subj_idx','rt','response']].copy()
    predictions['sample'] = 0
    predictions['response_sampled'] = 1
    predictions['rt_sampled'] = -predictions['rt']
    predictions = predictions.sample(frac=1,random_state=17).reset_index(drop=True)
    output = attach_predictive_outcomes(predictions, obs, 'addm', 1)
    assert len(output) == len(obs)
    assert (output.choice_group == 'S').all()
    assert (output.rt_value == predictions.rt).all()
    assert (output.dwell_quintile.to_numpy() ==
            predictions.merge(obs[TRIAL_KEYS+['dwell_quintile']],on=TRIAL_KEYS).dwell_quintile.to_numpy()).all()
    predictions.loc[0, 'trial'] = 999
    with pytest.raises(ValueError,match='match fitted trial|Duplicate trial'):
        attach_predictive_outcomes(predictions,obs,'addm',1)


def test_compact_comparison_tables():
    # Metrics are synthetic solely to validate table logic, not simulated HDDM.
    empirical = pd.DataFrame([
        dict(metric='accuracy', group='Overall', quintile=None, value=.8),
        *[dict(metric='p_choose_s_by_rt_quintile',group='S',quintile=i,value=.1*i)
          for i in range(1,6)],
    ])
    draws = []
    for draw in range(5):
        for row in empirical.to_dict('records'):
            draws.append(dict(**{k:v for k,v in row.items() if k!='value'},
                              value=row['value']+.01*draw))
    core, curves=summarise_metrics(empirical,pd.DataFrame(draws))
    assert len(core)==1 and len(curves)==5
    assert core.loc[0,'empirical']==.8
    assert core.loc[0,'ppc_mean']==pytest.approx(.82)
