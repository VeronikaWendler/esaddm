"""Pure parameter-recovery tests; real HDDM fitting is tested on BlueBEAR."""
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from esaddm import recovery as r


def test_contrast_algebra_preserves_es_definition():
    p = dict(b_a=np.array([2., 3.]), b_i=np.array([1., .8]),
             delta_a=np.array([.8, -.4]), delta_i=np.array([.6, .2]))
    d = r.parameter_dict(p)
    assert np.allclose(d['b_as'], [2.4, 2.8])
    assert np.allclose(d['b_ae'], [1.6, 3.2])
    assert np.allclose(d['b_is'], [1.3, .9])
    assert np.allclose(d['b_ie'], [.7, .7])
    assert np.allclose(d['theta_s'], d['b_is'] / d['b_as'])


def test_starting_point_uses_direct_z_no_double_logit():
    df = pd.DataFrame({'z': [.42, .63], 'z_trans': [-.32, .53],
                       'z_subj.2': [.39, .54], 'z_subj_trans.2': [-.45, .16]})
    assert np.allclose(r.trace_lookup(df, 'z', None), [.42, .63])
    assert np.allclose(r.trace_lookup(df, 'z', 2), [.39, .54])
    assert np.allclose(r.trace_lookup(df.drop(columns=['z']), 'z', None),
                       1 / (1 + np.exp([.32, -.53])))
    assert np.allclose(r.trace_lookup(df.drop(columns=['z_subj.2']), 'z', 2),
                       1 / (1 + np.exp([.45, -.16])))


def test_coherent_group_and_subject_joint_draw():
    df = {}
    for node in r.GROUP_NODES.values():
        df[node] = np.arange(60, dtype=float) / 100 + (0.2 if node == 'z' else 2)
        for sub in (2, 3):
            df[f'{node}_subj.{sub}'] = np.arange(60, dtype=float) / 100 + (.25 if node == 'z' else sub)
    frame = pd.DataFrame(df)
    model = types.SimpleNamespace(get_traces=lambda: frame)
    group, subj, idx = r.get_truth(model, [2, 3], seed=19)
    assert group['a'] == pytest.approx(2 + idx/100)
    assert subj[3]['a'] == pytest.approx(3 + idx/100)
    assert group['z'] == pytest.approx(.2 + idx/100)
    assert subj[2]['z'] == pytest.approx(.25 + idx/100)
    assert 'b_ae' in subj[2]


def test_trial_generator_uses_hddm_simulator_model_and_regression(monkeypatch):
    config = types.SimpleNamespace(model_config={r.SIM_MODEL: {'params':
                        ['v', 'a', 'z', 't', 'sz', 'sv', 'st']}})
    monkeypatch.setitem(sys.modules, 'hddm', types.ModuleType('hddm'))
    monkeypatch.setitem(sys.modules, 'hddm.model_config', config)
    design = pd.DataFrame({'subj_idx':[1, 2], 'attention_w':[.5, 1.],
                           'inattention_w':[.2, -.1],
                           'attention_contrast':[.1, .3],
                           'inattention_contrast':[.0, .1]})
    base = dict(a=1.8,t=.32,z=.55,b_a=2,b_i=1,delta_a=.5,delta_i=-.1)
    truth = {1:base, 2:base}
    theta = r.make_trial_parameters(design, truth)
    expected_drift = design[list(r.REGRESSORS)].to_numpy().dot([2,1,.5,-.1])
    assert theta.shape == (2,7)
    assert np.allclose(theta[:,0], expected_drift)
    assert np.allclose(theta[:,3], [.32,.32])
    assert np.all(theta[:,4:] == 0)


def test_synthetic_outlier_and_min_rt_without_any_real_hddm(monkeypatch):
    design = pd.DataFrame({'subj_idx':[1]*12, 'attention_w':[1.]*12,
                           'inattention_w':[0.]*12, 'attention_contrast':[0.]*12,
                           'inattention_contrast':[0.]*12})
    monkeypatch.setattr(r, 'make_trial_parameters', lambda *_: np.zeros((12,7)))
    monkeypatch.setattr(r, '_draw_ddm', lambda th: (np.full(len(th), .51),
                                                  np.arange(len(th))%2))
    frame, outliers, rounds = r.generate_synthetic(design, {1:{}}, seed=20,
                                                  min_rt=.25, p_outlier=0)
    assert len(frame)==12 and outliers == 0 and rounds == 0
    assert set(frame.response) == {0,1}
    assert (frame.rt >= .25).all()


def test_recovery_direct_and_derived_summaries():
    rng = np.random.default_rng(100)
    group_truth = dict(a=1.8,t=.4,z=.55,b_a=2,b_i=.9,delta_a=.4,delta_i=.25)
    group_truth.update({k:float(v) for k,v in r.parameter_dict(group_truth).items()})
    sub_truth = {1:dict(group_truth)}
    trace_mats=[]
    for _ in range(4):
        mat={}
        for k in r.DIRECT:
            v = group_truth[k]
            vals = rng.normal(v, .03 if k != 'z' else .008, size=250)
            mat['group__'+k] = vals
            mat['subj1__'+k] = vals + .001
        trace_mats.append(mat)
    summary, quality = r.summarise_chains(trace_mats, group_truth,sub_truth,[1],'study1',0)
    assert len(summary) == 2*(len(r.DIRECT)+len(r.DERIVED))
    assert set(summary.parameter)==set((*r.DIRECT,*r.DERIVED))
    assert summary.n_posterior_draws.eq(1000).all()
    assert summary.hdi_lower.le(summary.hdi_upper).all()
    assert summary.r_hat.notna().all()
    assert isinstance(quality,bool)


def test_invalid_group_trace_missing_fails_loudly():
    frame = pd.DataFrame({'a_std':[.1,.1], 'a_subj.1':[1,1]})
    with pytest.raises(KeyError):
        r.trace_lookup(frame,'a',None)


def test_one_recovery_replicate_plumbing_saves_only_small_outputs(monkeypatch, tmp_path):
    """Mocked plumbing only; real HDDM sampling must be piloted on BlueBEAR."""
    import json
    spec = types.SimpleNamespace(
        coordinate='s_upper_e_lower', estimate_z=True,
        predictors=r.REGRESSORS,
        drift_formula='v ~ 0 + attention_w + inattention_w + attention_contrast + inattention_contrast')
    monkeypatch.setattr(r, 'load_models', lambda root: {r.MODEL: spec})
    design = pd.DataFrame({'subj_idx': [1]*12+[2]*12,
                           'attention_w': [1.]*24, 'inattention_w':[0.]*24,
                           'attention_contrast':[.1]*24, 'inattention_contrast':[.1]*24})
    monkeypatch.setattr(r, 'check_input', lambda *args: (design, {'rows':24, 'participants':2}))
    cfg = tmp_path / 'config'; cfg.mkdir()
    (cfg/'study1.json').write_text(json.dumps({'min_rt_seconds':.25}))
    src = tmp_path/'studies'/'study1'/'results'/'fits'/'esaddm_z'
    src.mkdir(parents=True)
    (src/'esaddm_z_study1_run.json').write_text(json.dumps({
        'status':'completed','study':'study1','model':'esaddm_z','chains':[1,2,3,4,5]}))
    (src/'esaddm_z_study1_chain1.hddm').write_text('mock')
    hddm = types.ModuleType('hddm'); hddm.load=lambda path: object()
    monkeypatch.setitem(sys.modules, 'hddm', hddm)
    truth = dict(a=1.5, t=.4, z=.5, b_a=2.,b_i=.8, delta_a=.4, delta_i=.2)
    truth.update({k:float(v) for k,v in r.parameter_dict(truth).items()})
    monkeypatch.setattr(r, 'get_truth', lambda model, subjects, seed: (truth, {1:truth,2:truth}, 17))
    monkeypatch.setattr(r, 'generate_synthetic', lambda frame, truth_subj, seed, min_rt: (
        frame.assign(rt=.7, response=1)[['subj_idx','rt','response',*r.REGRESSORS]],1,0))
    def fake_fit(study, rep, chain, synthetic_csv, out, subjects, samples, burn, seed, spec, root_str):
        rng = np.random.default_rng(seed)
        arrays = {}
        for k in r.DIRECT:
            arrays['group__'+k]=rng.normal(truth[k], .02, samples-burn)
            for subj in subjects:
                arrays[f'subj{subj}__{k}']=rng.normal(truth[k], .02, samples-burn)
        np.savez_compressed(out, **arrays)
    monkeypatch.setattr(r, 'fit_chain', fake_fit)
    folder = r.run_rep('study1',0,root=tmp_path,reps=1,chains=4,samples=240,burn=40,jobs=1)
    assert (folder/'complete.json').exists()
    assert (folder/'synthetic.csv.gz').exists()
    assert (folder/'truth.json').exists()
    assert not (folder/'temporary_chains').exists()
    table=pd.read_csv(folder/'recovery_parameters.csv')
    assert len(table)==3*(len(r.DIRECT)+len(r.DERIVED))
    assert table['subj_idx'].isna().sum()==len(r.DIRECT)+len(r.DERIVED)
    assert r.run_rep('study1',0,root=tmp_path,reps=1,chains=4,samples=240,burn=40,jobs=1) == folder
    with pytest.raises(FileExistsError):
        r.run_rep('study1',0,root=tmp_path,reps=1,chains=4,samples=260,burn=40,jobs=1)
