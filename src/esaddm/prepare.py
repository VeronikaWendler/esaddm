"""ES identity and ES/EE accuracy model inputs
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

DECIMALS = 3
BASE_COLUMNS = [
    'subj_idx','rt','response','attention_w','inattention_w','phase','trial','sub_id',
    'cho','better_option','p_left','p_right','v_high','v_low','g_high','g_low',
    'ov_category','abs_category',
]
IDENTITY_COLUMNS = [
    'subj_idx','rt','response','phase','trial','sub_id','cho','op1','op2',
    'chosen_identity','p_left','p_right','g_left','g_right','v_e','v_s',
    'g_e','g_s','value_tie','attended_s','attended_e','unattended_s',
    'unattended_e','attention_w','inattention_w','attention_contrast',
    'inattention_contrast',
]


def _numeric(df, fields):
    for name in fields:
        df[name] = pd.to_numeric(df[name], errors='coerce')


def _validate_source(df):
    required = {'sub_id','phase','trial','cho','op1','op2','p1','p2',
                'DwellLeft','DwellRight','DwellTotal','rtime'}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f'AnalysisReady is missing required fields: {sorted(missing)}')
    if 'coordinate_system' in df:
        labels = df.loc[df.phase.isin(['LE','ES','EE']), 'coordinate_system']
        bad = [s for s in labels.dropna().astype(str).str.lower().unique()
               if 'physical' not in s]
        if bad:
            raise ValueError(f'Expected physical left/right coordinates, saw {bad}')
    es = df[df.phase.eq('ES')]
    if not ((es.op1.eq('E') & es.op2.eq('S')) |
            (es.op1.eq('S') & es.op2.eq('E'))).all():
        raise ValueError('Each ES trial must contain exactly one E and one S option')
    if len(es) and not 0.02 <= es.op1.eq('E').mean() <= 0.98:
        raise ValueError('ES options appear restructured instead of physical left/right')
    if 'chose_right_spatial' in df:
        cho = pd.to_numeric(df.cho, errors='coerce')
        right = pd.to_numeric(df.chose_right_spatial, errors='coerce')
        mask = df.phase.isin(['LE','ES','EE']) & cho.isin([1,2]) & right.notna()
        if not np.array_equal(cho[mask].eq(2).astype(int), right[mask].astype(int)):
            raise ValueError('Spatial choice coordinate is inconsistent')
    p = pd.concat([pd.to_numeric(df.p1,errors='coerce'),
                   pd.to_numeric(df.p2,errors='coerce')]).dropna()
    if p.empty or p.min() < -1e-9 or p.max() > 1 + 1e-9:
        raise ValueError('Probabilities must be in 0..1 (not 0..100)')


def _physical_features(df):
    z = df.copy()
    _numeric(z, ['p1','p2','cho','DwellLeft','DwellRight','DwellTotal','rtime'])
    z['p_left'] = z.p1.clip(0,1).round(DECIMALS)
    z['p_right'] = z.p2.clip(0,1).round(DECIMALS)
    total = z.DwellTotal.where(z.DwellTotal > 0)
    z['g_left'] = (z.DwellLeft / total).clip(0,1).round(DECIMALS)
    z['g_right'] = (z.DwellRight / total).clip(0,1).round(DECIMALS)
    z['rt'] = z.rtime.round(DECIMALS)
    z['subj_idx'] = pd.to_numeric(z.sub_id,errors='coerce')
    return z


def prepare_identity_es(raw: pd.DataFrame, exclusions: list[int], min_rt: float=0.250):
    """S upper (=1), E lower (=0), ties retained, RT >= 250ms."""
    _validate_source(raw)
    z = raw.loc[raw.phase.eq('ES')].copy()
    phase_n = len(z)
    _numeric(z,['sub_id','trial','cho','p1','p2','DwellLeft','DwellRight','DwellTotal','rtime'])
    z = z.loc[~z.sub_id.isin(exclusions)].copy()
    post_exclusion_n = len(z)
    z = _physical_features(z)
    e_left, s_left = z.op1.eq('E'), z.op1.eq('S')
    z['v_e'] = np.where(e_left, z.p_left, z.p_right)
    z['v_s'] = np.where(s_left, z.p_left, z.p_right)
    z['g_e'] = np.where(e_left, z.g_left, z.g_right)
    z['g_s'] = np.where(s_left, z.g_left, z.g_right)
    for col in ['v_e','v_s','g_e','g_s']:
        z[col] = pd.to_numeric(z[col],errors='coerce').round(DECIMALS)
    z['chosen_identity'] = pd.Series(index=z.index,dtype='object')
    z.loc[z.cho.eq(1), 'chosen_identity'] = z.loc[z.cho.eq(1),'op1']
    z.loc[z.cho.eq(2), 'chosen_identity'] = z.loc[z.cho.eq(2),'op2']
    z['response'] = np.nan
    valid_choice = z.chosen_identity.isin(['E','S'])
    z.loc[valid_choice,'response'] = z.loc[valid_choice,'chosen_identity'].eq('S').astype(int)
    z['attended_s'] = (z.g_s*z.v_s).round(DECIMALS)
    z['attended_e'] = (-z.g_e*z.v_e).round(DECIMALS)
    z['unattended_s'] = (z.g_e*z.v_s).round(DECIMALS)
    z['unattended_e'] = (-z.g_s*z.v_e).round(DECIMALS)
    z['attention_w'] = (z.attended_s+z.attended_e).round(DECIMALS)
    z['inattention_w'] = (z.unattended_s+z.unattended_e).round(DECIMALS)
    z['attention_contrast'] = ((z.attended_s-z.attended_e)/2).round(DECIMALS)
    z['inattention_contrast'] = ((z.unattended_s-z.unattended_e)/2).round(DECIMALS)
    z['value_tie'] = np.isclose(z.v_e,z.v_s,atol=1e-12,rtol=0)
    ties_before = int(z.value_tie.sum())
    before = len(z)
    z = z.loc[z.rt.notna() & (z.rt >= min_rt)].copy()
    rt_removed = before-len(z)
    needed = ['subj_idx','rt','response','v_e','v_s','g_e','g_s',
              'attention_w','inattention_w','attention_contrast','inattention_contrast']
    before = len(z)
    z = z.dropna(subset=needed).copy()
    missing_removed = before-len(z)
    if not z.response.isin([0,1]).all() or not np.allclose(z.g_e+z.g_s,1,atol=.002,rtol=0):
        raise ValueError('Invalid final S-upper response or E/S gaze proportions')
    audit = dict(phase='es', coordinate='s_upper_e_lower', ties_removed=0,
                 phase_rows=phase_n, rows_after_exclusions=post_exclusion_n,
                 ties_before_other_filters=ties_before, ties_retained=int(z.value_tie.sum()),
                 rows_removed_rt=rt_removed, rows_removed_missing=missing_removed,
                 final_rows=len(z), participants=int(z.subj_idx.nunique()),
                 s_choice_n=int(z.response.eq(1).sum()),e_choice_n=int(z.response.eq(0).sum()),
                 round_decimals=DECIMALS, min_rt=min_rt)
    return z[IDENTITY_COLUMNS].copy(),audit


def prepare_accuracy(raw: pd.DataFrame, phase: str, exclusions: list[int],
                     min_rt: float=0.250):
    """Higher-valued option upper (=1); ties removed; RT strictly > 250ms."""
    phase = phase.upper()
    if phase not in ('ES','EE'):
        raise ValueError('Accuracy-coded model only supports ES and EE')
    raw = raw.copy()
    raw['phase'] = raw.phase.astype(str).str.strip()
    _validate_source(raw)
    z = raw[raw.phase.eq(phase)].copy()
    phase_n = len(z)
    z = z.loc[~pd.to_numeric(z.sub_id,errors='coerce').isin(exclusions)].copy()
    after_exclusion_n = len(z)
    z = _physical_features(z)
    z['better_option'] = np.nan
    z.loc[z.p_left > z.p_right,'better_option'] = 1
    z.loc[z.p_right > z.p_left,'better_option'] = 2
    left_high = z.better_option.eq(1)
    right_high = z.better_option.eq(2)
    z['v_high'] = np.where(left_high,z.p_left,np.where(right_high,z.p_right,np.nan))
    z['v_low'] = np.where(left_high,z.p_right,np.where(right_high,z.p_left,np.nan))
    z['g_high'] = np.where(left_high,z.g_left,np.where(right_high,z.g_right,np.nan))
    z['g_low'] = np.where(left_high,z.g_right,np.where(right_high,z.g_left,np.nan))
    for col in ['v_high','v_low','g_high','g_low']:
        z[col] = z[col].round(DECIMALS)
    z['attention_w'] = (z.g_high*z.v_high-z.g_low*z.v_low).round(DECIMALS)
    z['inattention_w'] = (z.g_low*z.v_high-z.g_high*z.v_low).round(DECIMALS)
    z['response'] = np.nan
    valid_choice = z.cho.isin([1,2]) & z.better_option.isin([1,2])
    z.loc[valid_choice,'response'] = (
        z.loc[valid_choice,'cho'].astype(float) ==
        z.loc[valid_choice,'better_option'].astype(float)).astype(int)
    tie = z.p_left.notna() & z.p_right.notna() & np.isclose(z.p_left,z.p_right)
    n_ties = int(tie.sum())
    z = z.loc[~tie].copy()
    before = len(z)
    z = z.loc[z.rt > min_rt].copy()
    rt_removed = before-len(z)
    before = len(z)
    z = z.dropna(subset=['subj_idx','rt','response','attention_w','inattention_w']).copy()
    missing_removed = before-len(z)
    z['subj_idx'] = z.subj_idx.astype(int)
    z['response'] = z.response.astype(int)
    if not z.response.isin([0,1]).all():
        raise ValueError('Invalid accuracy-coded response')
    # These fields are metadata
    z = z.rename(columns={'OVcate_2':'ov_category','Abscate_2':'abs_category'})
    audit = dict(phase=phase.lower(),coordinate='higher_value_upper',
                 phase_rows=phase_n, rows_after_exclusions=after_exclusion_n,
                 ties_removed=n_ties,rows_removed_rt=rt_removed,
                 rows_removed_missing=missing_removed,
                 final_rows=len(z),participants=int(z.subj_idx.nunique()),
                 round_decimals=DECIMALS,min_rt=min_rt)
    return z[[col for col in BASE_COLUMNS if col in z.columns]].reset_index(drop=True),audit
