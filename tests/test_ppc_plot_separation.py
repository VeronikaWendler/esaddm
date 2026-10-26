"""Test plotting from saved summary draws (NO HDDM mocking / simulation)."""
import hashlib
import inspect
import json

import numpy as np
import pandas as pd
import pytest

from esaddm.ppc_plot import plot_boxplots, run_plotting


def _make_saved_stage(root, model):
    study = 'study1'
    dest = root / 'studies' / study / 'results' / 'ppc' / model
    data_dir = dest / 'data'
    data_dir.mkdir(parents=True)
    prefix = f'{model}_{study}'
    is_identity = model != 'accuracy_ee'
    groups = ('E', 'S') if is_identity else ('Correct', 'Error')
    observed = []
    simulated = []
    observed_subjects = []
    simulated_subjects = []

    core = [('mean_rt', 'Overall', None, .72),
            ('accuracy', 'Overall', None, .81)]
    for group in groups:
        core.append(('mean_rt', group, None, .7 if group in ('E', 'Correct') else .8))
        if is_identity:
            core.append(('accuracy', group, None, .79 if group == 'E' else .84))
    if is_identity:
        core.append(('p_choose_s', 'S', None, .58))
        for kind in ('p_choose_s_by_rt_quintile', 'p_choose_s_by_dwell_quintile'):
            for quintile in (1, 2, 3, 4, 5):
                core.append((kind, 'S', quintile, .25 + .11*quintile))
        core.extend([('p_choose_s_rt_quintile_slope', 'S', None, .11),
                     ('p_choose_s_dwell_quintile_slope', 'S', None, .11)])

    for metric, group, quintile, value in core:
        observed.append(dict(study=study, model=model, draw='observed', metric=metric,
                             group=group, quintile=quintile, value=value))
        for draw in range(6):
            simulated.append(dict(study=study, model=model, draw=f'1:batch1:1:{draw}',
                                  metric=metric, group=group, quintile=quintile,
                                  value=value + .004*(draw-3)))
    subject_measures = [('mean_rt','Overall'), ('accuracy','Overall')]
    subject_measures += [('mean_rt', group) for group in groups]
    if is_identity:
        subject_measures += [('accuracy', group) for group in groups]
    for subj_idx in range(1, 8):
        for metric, group in subject_measures:
            val = (.7 if metric == 'mean_rt' else .8) + .015*(subj_idx-4)
            observed_subjects.append(dict(draw='observed', subj_idx=subj_idx,
                                          metric=metric, group=group, value=val))
            for draw in range(6):
                simulated_subjects.append(dict(draw=f'1:batch1:1:{draw}', subj_idx=subj_idx,
                                               metric=metric, group=group,
                                               value=val + .006*(draw-3)))
    outputs = dict(observed_metrics=observed,
                   observed_subject_metrics=observed_subjects,
                   replicate_metrics=simulated,
                   replicate_subject_metrics=simulated_subjects)
    for name, rows in outputs.items():
        pd.DataFrame(rows).to_csv(data_dir / f'{prefix}_{name}.csv', index=False)
    (dest / f'{prefix}_ppc_run.json').write_text(json.dumps({
        'status': 'completed', 'study': study, 'model': model,
        'ppc_draws_generated': 6,
    }), encoding='utf-8')
    return dest, data_dir


@pytest.mark.parametrize('model,num_figures', [('esaddm_z',4), ('accuracy_ee',2)])
def test_tables_and_figures_rerun_without_reloading_simulator(tmp_path, model, num_figures):
    dest, raw = _make_saved_stage(tmp_path, model)
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in raw.iterdir()}
    run_plotting('study1', model, root=tmp_path)
    assert len(list((dest / 'plots').glob('*.png'))) == num_figures
    assert len(list((dest / 'plots').glob('*.pdf'))) == num_figures
    assert len(list(dest.glob('*.csv'))) == 3
    assert (dest / f'{model}_study1_ppc_plot_run.json').is_file()
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in raw.iterdir()}
    assert before == after
    # Re-running plots never touches the costly generated data or fails on existing images.
    run_plotting('study1', model, root=tmp_path)
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in raw.iterdir()}


def test_refuses_incomplete_ppc_data(tmp_path):
    dest, raw = _make_saved_stage(tmp_path, 'esaddm_z')
    manifest_path = dest / 'esaddm_z_study1_ppc_run.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['status'] = 'running'
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='not completed'):
        run_plotting('study1', 'esaddm_z', root=tmp_path)
    assert not (dest / 'plots').exists()


def test_boxplot_contains_only_boxes_no_participant_dots_or_connections():
    source = inspect.getsource(plot_boxplots)
    assert 'ax.scatter(' not in source
    assert 'ax.plot(' not in source
    assert 'showfliers=False' in source


def test_generation_saves_data_but_does_not_make_plots_or_tables(tmp_path, monkeypatch):
    """Stub-only pipeline control test; not a substitute for real HDDM testing."""
    import sys
    from types import ModuleType, SimpleNamespace
    from esaddm import ppc

    data = pd.DataFrame({
        'sub_id': [1]*5+[2]*5, 'subj_idx': [1]*5+[2]*5,
        'phase': ['ES']*10, 'trial': list(range(1,6))*2,
        'rt': [.5,.6,.7,.8,.9] * 2,
        'response': [0,1,0,1,0] * 2,
    }, index=list(range(10,15)) + list(range(20,25)))
    class FakeModel:
        def __init__(self, data):
            self.data = data
        def iter_observeds(self):
            for subj in (1, 2):
                sub = self.data[self.data.subj_idx.eq(subj)]
                yield f'wfpt.{subj}', {'node': SimpleNamespace(value=sub[['rt']])}
    fake_model = FakeModel(data)

    def generate(model, samples, append_data):
        assert append_data is False
        rows = []
        for node in ('wfpt.1', 'wfpt.2'):
            for sample in range(samples):
                for pos in range(5):
                    rows.append((node, sample, pos, .5 + .03*pos, float(pos%2)))
        return pd.DataFrame(rows, columns=['node','sample','row','rt','response']).set_index(
            ['node','sample','row'])

    fake_hddm = ModuleType('hddm')
    fake_hddm.load = lambda path: fake_model
    fake_hddm.utils = SimpleNamespace(post_pred_gen=generate)
    monkeypatch.setitem(sys.modules, 'hddm', fake_hddm)
    monkeypatch.setattr(ppc, 'load_models', lambda root: {'addm': SimpleNamespace()})
    monkeypatch.setattr(ppc, 'load_fit_metadata', lambda *args: ([tmp_path / 'fake.hddm'], {}))
    monkeypatch.setattr(ppc, 'check_input', lambda *args: (data, {}))
    def prepared_obs(*args):
        obs = data.reset_index(drop=True).copy()
        obs['better_source'] = 'S'
        obs['worse_source'] = 'E'
        obs['is_tie'] = False
        obs['dwell_quintile'] = [1,2,3,4,5]*2
        obs['choice_group'] = np.where(obs.response.eq(1), 'S', 'E')
        obs['accuracy_value'] = obs.response.astype(float)
        obs['rt_value'] = obs.rt.copy()
        obs['rt_quintile'] = ppc.rank_quintile(obs.rt)
        return obs
    monkeypatch.setattr(ppc, 'prepare_observed', prepared_obs)
    monkeypatch.setattr(ppc, 'verify_fit_observed', lambda a, b: a)
    output = ppc.run_ppc('study1', 'addm', chains=1, ppc_draws=5, batch_size=2,
                         root=tmp_path)
    assert not (output / 'plots').exists()
    assert list(output.glob('*.csv')) == []
    assert len(list((output / 'data').glob('*.csv'))) == 4
    assert len(list((output / 'data' / 'trials').glob('*.csv.gz'))) == 2
    assert len(pd.read_csv(output / 'data' / 'trials' / 'addm_study1_simulated_chain1.csv.gz')) == 50
    manifest = json.loads((output / 'addm_study1_ppc_run.json').read_text())
    assert manifest['status'] == 'completed'
    assert manifest['ppc_draws_generated'] == 5
    assert pd.read_csv(output / 'data' / 'addm_study1_replicate_subject_metrics.csv').draw.nunique() == 5


def test_legacy_completed_smoke_can_be_replotted_from_existing_tables(tmp_path):
    import shutil
    dest, raw = _make_saved_stage(tmp_path, 'esaddm_z')
    run_plotting('study1', 'esaddm_z', root=tmp_path)
    shutil.rmtree(raw)   # Prior pipeline didn't retain this source-data folder.
    run_plotting('study1', 'esaddm_z', root=tmp_path)
    assert len(list((dest / 'plots').glob('*.png'))) == 4
    assert len(list(dest.glob('*_summary.csv'))) >= 1
