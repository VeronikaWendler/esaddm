"""Run with python -m pytest -q"""
import json
from pathlib import Path
import pandas as pd
import pytest
from esaddm.cli import ROOT, file_sha, _canonical_bytes
from esaddm.prepare import prepare_accuracy, prepare_identity_es


def test_source_coordinates_and_tie_filtering():
    df = pd.DataFrame([
        # Value tie: retained for S-upper, removed for accuracy.
        dict(sub_id=7,phase='ES',trial=1,cho=2,op1='E',op2='S',p1=.5,p2=.5,
             DwellLeft=600,DwellRight=400,DwellTotal=1000,rtime=.45),
        # S on left, chosen left: response 1 for S-upper, 1 for accuracy (higher S).
        dict(sub_id=7,phase='ES',trial=2,cho=1,op1='S',op2='E',p1=.8,p2=.3,
             DwellLeft=700,DwellRight=300,DwellTotal=1000,rtime=.50),
        # E on left, chosen S: S-upper response 1, accuracy response 0 (S lower).
        dict(sub_id=7,phase='ES',trial=3,cho=2,op1='E',op2='S',p1=.9,p2=.2,
             DwellLeft=500,DwellRight=500,DwellTotal=1000,rtime=.60),
    ])
    identity, _ = prepare_identity_es(df, [], min_rt=.25)
    accuracy, _ = prepare_accuracy(df, 'ES', [], min_rt=.25)
    assert len(identity)==3 and len(accuracy)==2
    assert identity.response.tolist()==[1,1,1]
    assert accuracy.response.tolist()==[1,0]
    assert identity.value_tie.tolist()==[True,False,False]
    assert identity.loc[1,'v_s']==.8 and identity.loc[1,'v_e']==.3
    assert identity.loc[2,'attention_w']==round(.5*.2-.5*.9,3)


def test_rt_boundary_is_preserved():
    df = pd.DataFrame([
        dict(sub_id=7,phase='ES',trial=1,cho=1,op1='E',op2='S',p1=.8,p2=.2,
             DwellLeft=500,DwellRight=500,DwellTotal=1000,rtime=.250),
        dict(sub_id=7,phase='ES',trial=2,cho=2,op1='S',op2='E',p1=.8,p2=.2,
             DwellLeft=500,DwellRight=500,DwellTotal=1000,rtime=.300),
    ])
    identity, _ = prepare_identity_es(df, [])
    accuracy, _ = prepare_accuracy(df,'ES',[])
    assert len(identity)==2
    assert len(accuracy)==1


def test_specifications_are_no_intercept():
    models = json.loads((ROOT/'config/models.json').read_text())
    assert set(models)-{'note'} == {
        'accuracy_es','accuracy_ee','addm','addm_z','esaddm','esaddm_z'}
    for name,spec in models.items():
        if name=='note':
            continue
        assert spec['intercept'] is False
        assert spec['drift_formula'].startswith('v ~ 0 + ')
        assert 'attention_w' in spec['drift_formula']
        if name.startswith('esaddm'):
            assert 'attention_contrast' in spec['drift_formula']


@pytest.mark.parametrize('study',['study1','study2'])
@pytest.mark.parametrize('target',['identity_es','accuracy_es','accuracy_ee'])
def test_complete_original_data_matches_legacy_fingerprint(study,target):
    config = json.loads((ROOT/'config'/f'{study}.json').read_text())
    manifest = json.loads((ROOT/'verification/legacy_fingerprints.json').read_text())
    raw_path = ROOT/'studies'/study/'data'/'raw'/config['raw_file']
    if not raw_path.exists():
        pytest.skip('Raw AnalysisReady CSV not packaged in code-only distribution')
    assert file_sha(raw_path) == manifest[study]['raw_sha256']
    raw = pd.read_csv(raw_path,low_memory=False)
    if target=='identity_es':
        df, audit = prepare_identity_es(raw,config['excluded_subjects'])
    else:
        df, audit = prepare_accuracy(raw,target[-2:],config['excluded_subjects'])
    assert len(df) == config['expected_rows'][target]
    assert df.subj_idx.nunique() == config['expected_participants']
    import hashlib
    assert hashlib.sha256(_canonical_bytes(df)).hexdigest() == (
        manifest[study][target]['canonical_sha256'])
