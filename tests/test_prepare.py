""" data-preparation checks"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from esaddm.cli import ROOT, make_inputs
from esaddm.prepare import prepare_accuracy, prepare_identity_es


def test_boundary_coding_and_value_ties():
    trials = pd.DataFrame([
        dict(sub_id=7, phase='ES', trial=1, cho=2, op1='E', op2='S', p1=.5, p2=.5,
             DwellLeft=600, DwellRight=400, DwellTotal=1000, rtime=.45),
        dict(sub_id=7, phase='ES', trial=2, cho=1, op1='S', op2='E', p1=.8, p2=.3,
             DwellLeft=700, DwellRight=300, DwellTotal=1000, rtime=.50),
        dict(sub_id=7, phase='ES', trial=3, cho=2, op1='E', op2='S', p1=.9, p2=.2,
             DwellLeft=500, DwellRight=500, DwellTotal=1000, rtime=.60),
    ])
    identity, _ = prepare_identity_es(trials, [], min_rt=.25)
    accuracy, _ = prepare_accuracy(trials, 'ES', [], min_rt=.25)
    assert len(identity) == 3 and len(accuracy) == 2
    assert identity.response.tolist() == [1, 1, 1]
    assert accuracy.response.tolist() == [1, 0]
    assert identity.value_tie.tolist() == [True, False, False]


def test_rt_cutoff_is_phase_specific():
    trials = pd.DataFrame([
        dict(sub_id=7, phase='ES', trial=1, cho=1, op1='E', op2='S', p1=.8, p2=.2,
             DwellLeft=500, DwellRight=500, DwellTotal=1000, rtime=.250),
        dict(sub_id=7, phase='ES', trial=2, cho=2, op1='S', op2='E', p1=.8, p2=.2,
             DwellLeft=500, DwellRight=500, DwellTotal=1000, rtime=.300),
    ])
    identity, _ = prepare_identity_es(trials, [])
    accuracy, _ = prepare_accuracy(trials, 'ES', [])
    assert len(identity) == 2 and len(accuracy) == 1


@pytest.mark.parametrize('study', ['study1', 'study2'])
def test_real_input_preparation(study):
    config = json.loads((ROOT / 'config' / f'{study}.json').read_text())
    raw_path = ROOT / 'studies' / study / 'data' / 'raw' / config['raw_file']
    if not raw_path.is_file():
        pytest.skip('Real research data are intentionally excluded from Git')
    data = make_inputs(pd.read_csv(raw_path, low_memory=False), config)
    for name, frame in data.items():
        assert len(frame) == config['expected_rows'][name]
        assert frame.subj_idx.nunique() == config['expected_participants']
