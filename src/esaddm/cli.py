""" data preparation"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd
from .prepare import prepare_accuracy, prepare_identity_es

ROOT = Path(__file__).resolve().parents[2]
TARGETS = ('identity_es','accuracy_es','accuracy_ee')


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(frame):
    return frame.to_csv(index=False,float_format='%.3f').encode('utf-8')


def prepare_study(study: str, action: str, force: bool=False) -> None:
    config_path = ROOT/'config'/f'{study}.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    raw_path = ROOT/'studies'/study/'data'/'raw'/config['raw_file']
    output = ROOT/'studies'/study/'data'/'prepared'
    legacy_manifest = json.loads((ROOT/'verification'/'legacy_fingerprints.json').read_text())
    if not raw_path.is_file():
        raise FileNotFoundError(f'Missing {raw_path}; see README (copy original file unchanged)')
    original_sha = file_sha(raw_path)
    reference_sha = legacy_manifest[study]['raw_sha256']
    if original_sha != reference_sha:
        raise ValueError(f'{study}: raw input SHA mismatch. Expected {reference_sha}; got {original_sha}')
    raw = pd.read_csv(raw_path,low_memory=False)
    prepared = {}
    for target in TARGETS:
        if target == 'identity_es':
            data,audit = prepare_identity_es(raw,config['excluded_subjects'],config['min_rt_seconds'])
        else:
            data,audit = prepare_accuracy(raw,target[-2:],config['excluded_subjects'],config['min_rt_seconds'])
        if len(data) != config['expected_rows'][target]:
            raise ValueError(f'{study}/{target}: unexpected row count {len(data)}')
        if data.subj_idx.nunique() != config['expected_participants']:
            raise ValueError(f'{study}/{target}: unexpected number of participants')
        content = _canonical_bytes(data)
        actual = hashlib.sha256(content).hexdigest()
        expected = legacy_manifest[study][target]['canonical_sha256']
        if actual != expected:
            raise ValueError(f'{study}/{target}: legacy parity FAILED: {actual} != {expected}')
        prepared[target] = (content,dict(audit,raw_sha256=original_sha,
                         legacy_canonical_sha256=expected,config_sha256=file_sha(config_path)))
        print(f'PASS {study}/{target}: {len(data)} trials, '
              f'{data.subj_idx.nunique()} participants, legacy SHA-256 matched')
    if action == 'verify':
        # If a prepared file exists, verify that its content is still unmodified.
        for target,(content,_) in prepared.items():
            path = output/f'{target}.csv'
            if path.exists() and path.read_bytes() != content:
                raise ValueError(f'{path}: saved CSV does not match verified generated input')
        return
    if action != 'prepare':
        raise ValueError(f'Unknown action: {action}')
    existing = [str(output/f'{target}.csv') for target in TARGETS
                if (output/f'{target}.csv').exists()]
    if existing and not force:
        raise FileExistsError('Prepared files exist. Use verify; to replace deliberately use '
                              f'--force. Existing: {existing}')
    output.mkdir(parents=True,exist_ok=True)
    for target,(content,audit) in prepared.items():
        (output/f'{target}.csv').write_bytes(content)
        (output/f'{target}.audit.json').write_text(
            json.dumps(audit,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(f'Prepared 3 datasets in {output}')


def main() -> None:
    parser = argparse.ArgumentParser(description='ESaDDM reproducible preparation, no fitting')
    parser.add_argument('action',choices=['prepare','verify'])
    parser.add_argument('--study',choices=['study1','study2','both'],default='both')
    parser.add_argument('--force',action='store_true',help='Overwrite existing prepared CSVs')
    args = parser.parse_args()
    for study in (('study1','study2') if args.study=='both' else (args.study,)):
        prepare_study(study,args.action,force=args.force)

if __name__=='__main__':
    main()
