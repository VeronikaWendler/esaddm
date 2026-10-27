"""Tests for parameter-recovery calculations and file handling."""
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


def test_simulated_choices_and_rt_screening(monkeypatch):
    design = pd.DataFrame({'subj_idx':[1]*12, 'attention_w':[1.]*12,
                           'inattention_w':[0.]*12, 'attention_contrast':[0.]*12,
                           'inattention_contrast':[0.]*12})
    monkeypatch.setattr(r, 'make_trial_parameters', lambda *_: np.zeros((12,7)))
    monkeypatch.setattr(r, '_draw_ddm', lambda th: (np.full(len(th), .51),
                                                  np.arange(len(th))%2))
    frame, rounds = r.generate_synthetic(design, {1:{}}, seed=20,
                                         min_rt=.25)
    assert len(frame)==12 and rounds == 0
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
    """check recovery outputs with a small fitted-model stub."""
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
        frame.assign(rt=.7, response=1)[['subj_idx','rt','response',*r.REGRESSORS]],0))
    def fake_fit(study, rep, chain, synthetic_csv, out, subjects, samples, burn, seed, spec):
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


def test_parallel_chain_launcher_uses_independent_workers_and_limits_jobs(monkeypatch, tmp_path):
    """chain workers run as independent processes."""
    seen = []
    class FakeProcess:
        def __init__(self, command, cwd):
            self.command = command
            self.returncode = 0
            seen.append(command)
            chain = command[command.index('--worker-chain') + 1]
            (tmp_path / f'chain{chain}.npz').touch()
        def poll(self): return self.returncode
        def wait(self, timeout=None): return self.returncode
        def terminate(self): self.returncode = -15
        def kill(self): self.returncode = -9

    monkeypatch.setattr(r.subprocess, 'Popen', FakeProcess)
    monkeypatch.setattr(r.time, 'sleep', lambda _: None)
    remaining = [(i, str(tmp_path / f'chain{i}.npz')) for i in range(1, 5)]
    r.launch_independent_chain_processes('study1', 0, remaining, root=tmp_path,
                                         chains=4, samples=4000, burn=1000,
                                         seed=20261026, jobs=2)
    assert len(seen) == 4
    assert {cmd[cmd.index('--worker-chain')+1] for cmd in seen} == {'1','2','3','4'}
    assert all(cmd[:3] == [r.sys.executable, '-m', 'esaddm.recovery'] for cmd in seen)


def test_parallel_chain_launcher_fails_closed_on_worker_exit(monkeypatch, tmp_path):
    class FailedProcess:
        def __init__(self, command, cwd): self.returncode = 17
        def poll(self): return self.returncode
        def wait(self, timeout=None): return self.returncode
        def terminate(self): self.returncode = -15
        def kill(self): self.returncode = -9
    monkeypatch.setattr(r.subprocess, 'Popen', FailedProcess)
    remaining = [(1, str(tmp_path / 'chain1.npz'))]
    with pytest.raises(RuntimeError, match='failed'):
        r.launch_independent_chain_processes('study1', 0, remaining, root=tmp_path,
                                             chains=4, samples=4000, burn=1000,
                                             seed=20261026, jobs=4)
    assert not (tmp_path / 'chain1.npz').exists()


def test_theta_is_derived_drawwise_and_saved_with_recovery():
    """calculate theta from matched posterior draws."""
    p = dict(b_a=np.array([1., 3.]), b_i=np.array([.4, 1.2]),
             delta_a=np.array([.2, -.2]), delta_i=np.array([.1, .3]))
    d = r.parameter_dict(p)
    assert np.allclose(d['theta_s'], d['b_is'] / d['b_as'])
    assert np.allclose(d['theta_e'], d['b_ie'] / d['b_ae'])
    assert np.allclose(d['delta_theta_s_minus_e'], d['theta_s'] - d['theta_e'])
    # averaging ratios is not the same as dividing average coefficients
    assert not np.isclose(d['theta_s'].mean(), d['b_is'].mean() / d['b_as'].mean())


def test_undefined_ratio_is_flagged_without_breaking_primary_recovery():
    """flag unstable theta values while saving primary parameters."""
    rng = np.random.default_rng(322)
    p = dict(a=1.5, t=.32, z=.53, b_a=0., b_i=.8,
             delta_a=0., delta_i=.3)
    p.update({k: float(v) for k, v in r.parameter_dict(p).items()})
    traces = []
    for c in range(4):
        d = {}
        for k in r.DIRECT:
            # theta is undefined when the attended coefficient is zero
            x = (np.zeros(250) if k in ('b_a', 'delta_a') else
                 rng.normal(p[k], .01, size=250))
            d['group__'+k] = x
            d['subj1__'+k] = x.copy()
        traces.append(d)
    summary, _ = r.summarise_chains(traces, p, {1:p}, [1], 'study1', 0)
    ratios = summary.loc[summary.parameter.isin(r.RATIOS)]
    assert len(ratios) == 2*len(r.RATIOS)
    assert ratios.recovered_mean.isna().all()
    assert ratios.ratio_warning.str.contains('insufficient_defined_ratio_draws').all()
    assert summary.loc[summary.parameter.eq('a'), 'recovered_mean'].notna().all()


def test_recovery_mcmc_defaults_match_empirical_fit():
    assert r.DEFAULT_CHAINS == 5
    assert r.DEFAULT_SAMPLES == 5000
    assert r.DEFAULT_BURN == 1000
    assert r.DEFAULT_PARALLEL_JOBS == 4


def test_simulated_outcomes_match_hddm_simulator(monkeypatch):
    """simulated outcomes are returned unchanged when rt passes the cutoff."""
    design = pd.DataFrame({'subj_idx': [1] * 6,
                           'attention_w': [1.] * 6,
                           'inattention_w': [0.] * 6,
                           'attention_contrast': [0.] * 6,
                           'inattention_contrast': [0.] * 6})
    monkeypatch.setattr(r, 'make_trial_parameters', lambda *_: np.zeros((6, 7)))
    calls = []
    def sim(theta):
        calls.append(len(theta))
        return np.full(len(theta), 1.1), np.zeros(len(theta), dtype=int)
    monkeypatch.setattr(r, '_draw_ddm', sim)
    simulated, rounds = r.generate_synthetic(design, {1:{}}, seed=13, min_rt=.25)
    assert calls == [6]
    assert rounds == 0
    assert np.allclose(simulated.rt, 1.1)
    assert simulated.response.eq(0).all()
    assert r.VERSION == 2


def test_dataset_version_is_checked(monkeypatch, tmp_path):
    import json
    spec = types.SimpleNamespace(coordinate='s_upper_e_lower', estimate_z=True,
                                 predictors=r.REGRESSORS)
    monkeypatch.setattr(r, 'load_models', lambda root: {r.MODEL: spec})
    design = pd.DataFrame({'subj_idx': [1], 'rt': [.7], 'response': [1],
                           **{c: [1.] for c in r.REGRESSORS}})
    monkeypatch.setattr(r, 'check_input', lambda *args: (design, {'rows': 1}))
    monkeypatch.setitem(sys.modules, 'hddm', types.ModuleType('hddm'))
    config = tmp_path / 'config'
    config.mkdir()
    (config/'study1.json').write_text(json.dumps({'min_rt_seconds': .25}))
    rep = tmp_path/'studies/study1/results/recovery/esaddm_z/rep_00'
    rep.mkdir(parents=True)
    design.to_csv(rep/'synthetic.csv.gz', index=False, compression='gzip')
    (rep/'truth.json').write_text(json.dumps({'study':'study1','rep':0,'version':1,
                        'recovery_config': {'chains':5,'samples':5000,'burn':1000,'seed':20261026}}))
    with pytest.raises(ValueError, match='format version'):
        r.run_rep('study1', 0, root=tmp_path, reps=1, jobs=1)
