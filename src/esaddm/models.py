""" model registry and input checks shared by both studies"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    input_name: str
    coordinate: str
    drift_formula: str
    predictors: tuple[str, ...]
    estimate_z: bool

    @property
    def include(self) -> list[str]:
        return ['a', 't', 'v'] + (['z'] if self.estimate_z else [])


def load_models(root: Path = ROOT) -> dict[str, ModelSpec]:
    """Load the single authoritative model registry; reject unexpected formulas."""
    raw = json.loads((root / 'config/models.json').read_text(encoding='utf-8'))
    registry = {}
    for name, entry in raw.items():
        if name == 'note':
            continue
        formula = entry['drift_formula']
        if entry.get('intercept') is not False or not formula.startswith('v ~ 0 + '):
            raise ValueError(f'{name}: dont need intercept')
        predictors = tuple(part.strip() for part in formula.split('~', 1)[1][4:].split('+'))
        if not predictors or len(predictors) != len(set(predictors)):
            raise ValueError(f'{name}: invalid predictors in {formula}')
        if entry['coordinate'] not in {'higher_value_upper', 's_upper_e_lower'}:
            raise ValueError(f'{name}: invalid coordinate')
        if entry['input'] not in {'accuracy_es.csv', 'accuracy_ee.csv', 'identity_es.csv'}:
            raise ValueError(f'{name}: unexpected input file')
        registry[name] = ModelSpec(
            name, entry['input'], entry['coordinate'], formula,
            predictors, bool(entry['estimate_z']),
        )
    if set(registry) != {'accuracy_es', 'accuracy_ee', 'addm', 'addm_z', 'esaddm', 'esaddm_z'}:
        raise ValueError('Model registry must contain exactly six prespecified models')
    for name, spec in registry.items():
        if name.startswith('accuracy_') != (spec.coordinate == 'higher_value_upper'):
            raise ValueError(f'{name}: coordinate mismatch')
        expected = ('attention_w', 'inattention_w')
        if name.startswith('esaddm'):
            expected += ('attention_contrast', 'inattention_contrast')
        if spec.predictors != expected:
            raise ValueError(f'{name}: predictors changed: {spec.predictors}')
    return registry


def check_input(study: str, spec: ModelSpec, root: Path = ROOT) -> tuple[pd.DataFrame, dict]:
    """Validate response coordinate, participant inclusion, regressors and design rank.

    Checks never write files and never import HDDM. This is the first cluster gate.
    """
    if study not in {'study1', 'study2'}:
        raise ValueError('Study must be study1 or study2')
    config = json.loads((root / 'config' / f'{study}.json').read_text(encoding='utf-8'))
    path = root / 'studies' / study / 'data' / 'prepared' / spec.input_name
    if not path.is_file():
        raise FileNotFoundError(f'Missing prepared input: {path}')
    frame = pd.read_csv(path, low_memory=False)
    required = {'subj_idx', 'rt', 'response', 'phase', *spec.predictors}
    if spec.coordinate == 's_upper_e_lower':
        required |= {'v_s', 'v_e', 'g_s', 'g_e', 'chosen_identity',
                     'attended_s', 'attended_e', 'unattended_s', 'unattended_e',
                     'attention_w', 'inattention_w', 'attention_contrast', 'inattention_contrast'}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f'{study}/{spec.name}: missing columns: {sorted(missing)}')
    numeric = ['subj_idx', 'rt', 'response', *spec.predictors]
    numeric += (['v_s', 'v_e', 'g_s', 'g_e', 'attended_s',
                 'attended_e', 'unattended_s', 'unattended_e']
                if spec.coordinate == 's_upper_e_lower' else [])
    for field in numeric:
        frame[field] = pd.to_numeric(frame[field], errors='raise')
    if frame[numeric].isna().any().any():
        raise ValueError(f'{study}/{spec.name}: missing numerical input')
    if (frame.rt < config['min_rt_seconds'] - 1e-10).any():
        raise ValueError(f'{study}/{spec.name}: RT below required cutoff')
    if not frame.response.isin((0, 1)).all():
        raise ValueError(f'{study}/{spec.name}: response must be binary')
    if not np.allclose(frame.subj_idx, frame.subj_idx.astype(int)):
        raise ValueError('Participant IDs must be integer-valued')
    frame['subj_idx'] = frame.subj_idx.astype(int)
    frame['response'] = frame.response.astype(int)
    if set(frame.subj_idx).intersection(config['excluded_subjects']):
        raise ValueError('Excluded participant reappeared in prepared input')
    if frame.subj_idx.nunique() != config['expected_participants']:
        raise ValueError(f'{study}/{spec.name}: participant count mismatch')
    expected_rows = config['expected_rows'][spec.input_name.removesuffix('.csv')]
    if len(frame) != expected_rows:
        raise ValueError(f'{study}/{spec.name}: expected {expected_rows} rows, got {len(frame)}')
    phase = 'EE' if spec.input_name == 'accuracy_ee.csv' else 'ES'
    if set(frame.phase.astype(str).str.strip()) != {phase}:
        raise ValueError(f'{study}/{spec.name}: incorrect phase')

    if spec.coordinate == 's_upper_e_lower':
        if not (frame.chosen_identity.map({'S': 1, 'E': 0}) == frame.response).all():
            raise ValueError('S-upper choice coding is inconsistent')
        if not np.allclose(frame.g_e + frame.g_s, 1, atol=.002, rtol=0):
            raise ValueError('Dwell proportions must add to one')
        derived = {
            'attention_w': frame.attended_s + frame.attended_e,
            'inattention_w': frame.unattended_s + frame.unattended_e,
            'attention_contrast': (frame.attended_s - frame.attended_e) / 2,
            'inattention_contrast': (frame.unattended_s - frame.unattended_e) / 2,
        }
        for label, reconstruction in derived.items():
            if not np.allclose(frame[label], reconstruction, atol=.00201, rtol=0):
                raise ValueError(f'{study}/{spec.name}: invalid {label} regressor')
    else:
        if not (frame.rt > config['min_rt_seconds']).all():
            raise ValueError('Accuracy-coded RT filter must be strictly greater than cutoff')
        if {'v_high', 'v_low'}.issubset(frame):
            if not (frame.v_high > frame.v_low).all():
                raise ValueError('Accuracy-coded value-tie filtering failed')

    design = frame.loc[:, list(spec.predictors)].to_numpy(dtype=float)
    rank = int(np.linalg.matrix_rank(design))
    if rank != len(spec.predictors):
        raise ValueError(f'{study}/{spec.name}: design rank {rank} < {len(spec.predictors)}')
    report = {
        'study': study, 'model': spec.name, 'input': str(path.relative_to(root)),
        'rows': len(frame),
        'participants': int(frame.subj_idx.nunique()), 'coordinate': spec.coordinate,
        'formula': spec.drift_formula, 'estimated_parameters': spec.include,
        'design_rank': rank, 'n_predictors': len(spec.predictors),
        'n_response_1': int(frame.response.sum()),
        'n_response_0': int((1 - frame.response).sum()),
    }
    return frame, report


def build_hddm_model(data: pd.DataFrame, spec: ModelSpec, hddm_module=None):
    """Build but do not sample the original family-style HDDMRegressor."""
    if hddm_module is None:
        import hddm as hddm_module  # deferred: checks work without HDDM installed
    return hddm_module.HDDMRegressor(
        data,
        {'model': spec.drift_formula, 'link_func': lambda value: value},
        include=spec.include,
        p_outlier=0.05,
        group_only_regressors=False,
        keep_regressor_trace=True,
    )
