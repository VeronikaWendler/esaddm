"""Model design checks; actual HDDM sampling is tested with `smoke`."""
from __future__ import annotations

import pytest

from esaddm.models import ROOT, check_input, load_models


def test_all_models_have_no_drift_intercept():
    specs = load_models()
    assert len(specs) == 6
    for model in specs.values():
        assert model.drift_formula.startswith('v ~ 0 + ')
    assert specs['esaddm_z'].include == ['a', 't', 'v', 'z']
    assert specs['esaddm'].include == ['a', 't', 'v']
    assert specs['accuracy_ee'].coordinate == 'higher_value_upper'
    assert specs['esaddm_z'].coordinate == 's_upper_e_lower'


@pytest.mark.parametrize('study', ('study1', 'study2'))
@pytest.mark.parametrize('model_name', (
    'accuracy_es', 'accuracy_ee', 'addm', 'addm_z', 'esaddm', 'esaddm_z'))
def test_actual_prepared_data(study, model_name):
    spec = load_models()[model_name]
    path = ROOT / 'studies' / study / 'data' / 'prepared' / spec.input_name
    if not path.is_file():
        pytest.skip('Prepared research data are excluded from the code repository')
    data, report = check_input(study, spec)
    assert report['design_rank'] == report['n_predictors']
    assert len(data) == report['rows']
