"""Conservative recovery: freeze data/plan, reuse complete folds, restart incomplete folds.

Never claims epoch/checkpoint continuation. Original attempts remain immutable.
"""
from __future__ import annotations

__all__ = [
    'inspect_resume',
    'copy_completed_fold',
    'verify_current_labels',
]
from pathlib import Path
import json
import re
import shutil
import numpy as np
import pandas as pd
from .bundle import file_digest, load, write_json


def inspect_resume(root, results_root, run_id):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run_id):
        raise ValueError('Invalid resume run ID')
    root, results_root = Path(root), Path(results_root)
    source = results_root/run_id
    report = json.loads((source/'suite.json').read_text())
    plan = json.loads((source/'walk_forward.json').read_text())
    dataset = Path(report['dataset']).resolve()
    manifest = load(dataset)
    if manifest['version'] != report['data_version']:
        raise ValueError('Frozen dataset version changed')
    # Orchestration/mail can change; scientific input/model implementation cannot.
    excluded = {'pipeline.py', 'server_suite.py', 'training_mail.py', 'resume.py', '6_factor_sequence_verify.py'}
    checked = 0
    for relative, digest in report['code'].items():
        if Path(relative).name in excluded:
            continue
        path = (root/relative).resolve()
        if not path.is_relative_to(root.resolve()) or file_digest(path) != digest:
            raise ValueError(f'Scientific source changed: {relative}')
        checked += 1
    if not checked or not plan['folds']:
        raise ValueError('Missing frozen source/plan evidence')
    return source, report, plan, dataset


def copy_completed_fold(root, source, destination, split, data_version, research, ablations):
    """Return False for an unfinished fold; reject a supposedly completed but corrupt fold."""
    from .pipeline import read_predictions
    from src.res.algo.nn.model.factor_sequence import MODELS
    source, destination = Path(source), Path(destination)
    marker = source/'smoke.json'
    if not marker.exists() or json.loads(marker.read_text()).get('status') != 'passed':
        return False
    training = source/'training'
    if json.loads((training/'split.json').read_text()) != split:
        raise ValueError('Completed fold split differs from frozen plan')
    runs = json.loads((training/'runs.json').read_text())
    variants = [(key,{}) for key in MODELS]
    if ablations:
        variants += [('factorseq_m1',{'use_performance':False}),('factorseq_m1',{'use_market':False}),('factorseq_m2',{'modulation':False})]
    expected = {(model, seed, json.dumps(extra,sort_keys=True)) for model,extra in variants
                for seed in ((17,29,43) if research else (17,))}
    actual = {(r['model'],r['seed'],json.dumps(r.get('variant',{}),sort_keys=True)) for r in runs}
    if len(runs) != len(expected) or actual != expected:
        raise ValueError('Completed fold has an incomplete model set')
    for run in runs:
        if run['status'] != 'passed' or run['data_version'] != data_version:
            raise ValueError('Invalid completed training result')
        for relative, digest in run['provenance']['training_data_code'].items():
            path = (Path(root)/relative).resolve()
            if not path.is_relative_to(Path(root).resolve()) or file_digest(path) != digest:
                raise ValueError(f'Training data source changed: {relative}')
        pred = pd.read_csv(training/f"{run['name']}.csv")
        if sorted(pred.date.unique()) != split['test'] or pred.duplicated(['date','secid']).any():
            raise ValueError('Completed predictions have wrong dates/duplicates')
        columns = ['date','secid','pred','label']
        saved = read_predictions(run['model_path'])
        left = pred[columns].sort_values(columns[:2]).reset_index(drop=True)
        right = saved[columns].sort_values(columns[:2]).reset_index(drop=True)
        if not np.isfinite(left[['pred','label']]).all().all():
            raise ValueError('Nonfinite completed prediction')
        pd.testing.assert_frame_equal(left,right,check_dtype=False,rtol=1e-6,atol=1e-8)
    destination.mkdir(parents=True, exist_ok=False)
    # Copy result files only; never checkpoints or SMTP delivery ledgers.
    target = destination/'training'
    target.mkdir()
    for path in training.iterdir():
        if path.is_file() and path.suffix in ('.json','.csv','.npz','.log'):
            shutil.copy2(path,target/path.name)
    write_json(destination/'smoke.json',{'status':'passed','reused_from':str(source)})
    write_json(destination/'reuse.json',{'source':str(source),'files':{
        p.name:file_digest(p) for p in target.iterdir() if p.is_file()}})
    return True


def verify_current_labels(source, dataset):
    """Do not mix old predictions with silently revised training labels on resume."""
    snapshots=sorted(source.parent.glob(source.name+'_f*/training/labels.npz'))
    if not snapshots:
        return  # No prior training inputs exist to reuse.
    from src.data.util import DataBlock
    with np.load(snapshots[0],allow_pickle=False) as old:
        y=DataBlock.load_preprocess('y',frame='fit')
        y=y.align_feature(['std_lag1_10']).align_secid(old['secids']).align_date(old['dates'])
        current=y.values.cpu().numpy()
        if current.shape != old['values'].shape or not np.array_equal(current,old['values'],equal_nan=True):
            raise ValueError('Historical labels changed since original run; refusing mixed-version resume')
