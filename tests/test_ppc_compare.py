"""Offline four-model comparisons from stored trials/metrics; never run HDDM."""
import gzip
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from esaddm.ppc_compare import (
    MODELS, COLORS, load_sources, run_comparison,
    _median_from_observed, _median_from_simulated,
)


def make_fixture(root, study='study1'):
    rows = []
    for subj in (1, 2, 3):
        for trial in range(1, 11):
            chose_s = (trial + subj) % 2
            better = 'S' if trial % 3 else 'E'
            rows.append(dict(sub_id=subj, phase='ES', trial=trial, subj_idx=subj,
                             response=chose_s, rt=0.4 + trial*.04 + subj*.003,
                             rt_value=0.4 + trial*.04 + subj*.003,
                             choice_group='S' if chose_s else 'E',
                             accuracy_value=float(('S' if chose_s else 'E') == better),
                             is_tie=False, better_source=better,
                             dwell_ms_s_minus_e=trial * 22 - 110,
                             dwell_quintile=(trial - 1)//2 + 1))
    trials = pd.DataFrame(rows)
    observed = [dict(study=study, model='', draw='observed',
                     metric=metric, group=grp, quintile=np.nan, value=value)
                for metric, grp, value in [('mean_rt','Overall',.63), ('accuracy','Overall',.76),
                                           ('mean_rt','E',.65),('mean_rt','S',.66),
                                           ('accuracy','E',.72),('accuracy','S',.80),
                                           ('p_choose_s','S',.55),
                                           ('p_choose_s_rt_quintile_slope','S',.08),
                                           ('p_choose_s_dwell_quintile_slope','S',.12)]]
    for typ in ('p_choose_s_by_rt_quintile','p_choose_s_by_dwell_quintile'):
        for q in range(1,6):
            observed.append(dict(study=study, model='', draw='observed',
                                 metric=typ, group='S', quintile=q, value=.2 + q*.1))
    for modelidx, model in enumerate(MODELS):
        folder = root / 'studies' / study / 'results' / 'ppc' / model
        prefix = f'{model}_{study}'
        raw = folder / 'data'
        trialdir = raw / 'trials'
        trialdir.mkdir(parents=True, exist_ok=True)
        trials.to_csv(trialdir / f'{prefix}_observed_trials.csv.gz', index=False,
                      compression='gzip')
        obs_metric = pd.DataFrame(observed)
        obs_metric['model'] = model
        obs_metric.to_csv(raw / f'{prefix}_observed_metrics.csv', index=False)
        sim = []
        for ci in (1,2):
            simrows = []
            for j in range(3):
                draw = f'{ci}:batch1:{j}'
                one = trials[['sub_id','phase','trial','subj_idx','rt_value']].copy()
                one['ppc_draw'] = draw
                one['response_sampled'] = trials['response'].to_numpy()
                one['rt_value'] += .007 * (modelidx + j)
                simrows.append(one)
                for item in observed:
                    sim.append({**item, 'model':model, 'draw':draw,
                                'value':float(item['value'] + .005 * (modelidx + j))})
            df = pd.concat(simrows, ignore_index=True)
            df.to_csv(trialdir / f'{prefix}_simulated_chain{ci}.csv.gz',
                      compression='gzip', index=False)
        pd.DataFrame(sim).to_csv(raw / f'{prefix}_replicate_metrics.csv',index=False)
        (folder / f'{prefix}_ppc_run.json').write_text(json.dumps({
            'status':'completed','study':study,'model':model,'chains':2,
            'ppc_draws_generated':6,'draws_per_chain':[3,3],
        }))
    return trials


@pytest.mark.parametrize('study', ['study1','study2'])
def test_comparison_complete_and_cached(tmp_path, study):
    original = make_fixture(tmp_path, study)
    first = run_comparison(study, tmp_path)
    plots = sorted((first / 'plots').glob('*.png'))
    assert len(plots) == 5
    assert len(list((first / 'plots').glob('*.pdf'))) == 5
    summary = pd.read_csv(first / f'{study}_four_model_comparison_summary.csv')
    curves = pd.read_csv(first / f'{study}_four_model_comparison_quintiles.csv')
    ranking = pd.read_csv(first / f'{study}_four_model_model_rankings.csv')
    assert summary.model.nunique() == 4
    assert {'accuracy','mean_rt','median_rt'}.issubset(summary.metric.unique())
    assert len(curves.query("metric == 'p_choose_s_by_rt_quintile'")) == 20
    assert len(curves.query("metric == 'p_choose_s_by_dwell_quintile'")) == 20
    assert ranking.groupby('comparison').size().eq(4).all()
    assert ranking.groupby('comparison').rank_by_rmse.min().eq(1).all()
    med = summary.query("model == 'addm' and metric == 'median_rt' and group == 'Overall'")
    assert med.iloc[0].empirical == pytest.approx(
        original.groupby('subj_idx').rt_value.median().mean())
    before = {str(p):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in tmp_path.glob(f'studies/{study}/results/ppc/*/data/*.csv')}
    cache = first / 'data' / f'{study}_median_rt_by_draw.csv'
    created = cache.stat().st_mtime_ns
    second = run_comparison(study, tmp_path, show_bands=False)
    assert second == first and cache.stat().st_mtime_ns == created
    assert before == {str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in tmp_path.glob(f'studies/{study}/results/ppc/*/data/*.csv')}


def test_fails_instead_of_comparing_mismatched_observations(tmp_path):
    make_fixture(tmp_path)
    target = (tmp_path / 'studies/study1/results/ppc/esaddm_z/data/trials/'
              'esaddm_z_study1_observed_trials.csv.gz')
    df = pd.read_csv(target)
    df.loc[0, 'response'] = 1 - df.loc[0, 'response']
    df.to_csv(target, compression='gzip', index=False)
    with pytest.raises(ValueError, match='Choice coding|Empirical data mismatch'):
        run_comparison('study1', tmp_path)


def test_fails_instead_of_comparing_different_ppc_counts(tmp_path):
    make_fixture(tmp_path)
    target = tmp_path / 'studies/study1/results/ppc/addm/addm_study1_ppc_run.json'
    data = json.loads(target.read_text())
    data['ppc_draws_generated'] = 7
    target.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='Different PPC draw counts|Saved draw count'):
        run_comparison('study1', tmp_path)


def test_will_not_accept_incomplete_chain_data(tmp_path):
    make_fixture(tmp_path)
    target = (tmp_path / 'studies/study1/results/ppc/addm/data/trials/'
              'addm_study1_simulated_chain2.csv.gz')
    df = pd.read_csv(target)
    df = df.iloc[:-1]
    df.to_csv(target, compression='gzip', index=False)
    with pytest.raises(ValueError, match='Incorrect number of predicted trials'):
        run_comparison('study1', tmp_path)


def test_legacy_tab10_colours():
    assert COLORS == {'addm':'#1f77b4','addm_z':'#ff7f0e',
                      'esaddm':'#2ca02c','esaddm_z':'#d62728'}
