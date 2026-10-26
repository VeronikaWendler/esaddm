from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest

from esaddm.ppc import align_predictions_by_observed_node, attach_predictive_outcomes

class Stub:
    def __init__(self, data):
        self.data = data
    def iter_observeds(self):
        for name, subj_id in [('wfpt.1', 1), ('wfpt.2', 2)]:
            node_value = self.data[self.data.subj_idx.eq(subj_id)][['rt']]
            yield name, {'node': SimpleNamespace(value=node_value)}


def trial_data():
    return pd.DataFrame({
        'sub_id': [1,1,1,2,2,2], 'subj_idx': [1,1,1,2,2,2],
        'phase': ['ES']*6, 'trial': [1,2,3,1,2,3],
        'rt': [.5,.6,.7,.8,.9,1.0], 'response': [0,1,0,1,0,1],
        'better_source':['S']*6,'worse_source':['E']*6,
        'is_tie':[False]*6,'dwell_quintile':[1,2,3,4,5,1],
    }, index=[10,11,12,25,26,27])


def pred(local=True):
    # This is the actual HDDM documented (node,sample,row) MultiIndex shape.
    rows=[]
    for node in ('wfpt.1','wfpt.2'):
        for sample in range(2):
            for pos, label in enumerate(([10,11,12] if node=='wfpt.1' else [25,26,27])):
                rows.append((node,sample,pos if local else label,.5+sample,.0))
    return pd.DataFrame(rows, columns=['node','sample','row','rt','response']).set_index(['node','sample','row'])

@pytest.mark.parametrize('local',[True,False])
def test_reconstruct_full_draws_from_original_node_rows(local):
    model=Stub(trial_data())
    got=align_predictions_by_observed_node(pred(local=local),model)
    assert len(got)==12
    assert got.groupby('sample').size().tolist()==[6,6]
    assert not got.duplicated(['sample','sub_id','phase','trial']).any()
    obs=model.data.reset_index(drop=True)
    out=attach_predictive_outcomes(got,obs,'addm',1)
    assert len(out)==12
    assert sorted(out.ppc_draw.unique())==['1:0','1:1']


def test_reject_duplicate_likelihood_node_data():
    d=trial_data()
    class Overlap(Stub):
        def iter_observeds(self):
            for name in ('wfpt.1','wfpt.2'):
                yield name, {'node':SimpleNamespace(value=d[['rt']].iloc[:3])}
    with pytest.raises(ValueError,match='partition fitted trials'):
        align_predictions_by_observed_node(pred(), Overlap(d))


def test_reject_unknown_simulation_trial_indices():
    bad=pred().reset_index()
    bad.loc[bad['node'].eq('wfpt.2'),'row'] += 4
    with pytest.raises(ValueError,match='Unrecognised|does not contain'):
        align_predictions_by_observed_node(bad.set_index(['node','sample','row']),Stub(trial_data()))
