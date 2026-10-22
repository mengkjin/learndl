# description: Build isolated dd51 15m_original inputs, audit them, then request training
# content: Prefer authenticated legacy snapshots; otherwise recompute from minute sources with frozen dd51 formulas.
# autofix: true
# email: true
# mode: shell
# parameters:
#   dry_run:
#     type: bool
#     default: false
#     desc: Check sources and provenance without writing or requesting training
#   snapshot_manifest:
#     type: str
#     default: ''
#     desc: Optional authenticated historical snapshot manifest (see experiment documentation)
from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import torch

from src.proj import CALENDAR, DB, PATH, Load, Logger
from src.proj.util.script import ScriptTool
from src.data import DataBlock
from src.data.preprocess.processors.original_15m import BASELINE_COMMIT, PrePro_15m_original

KEY = '15m_original'
SCHEDULE = 'gru_15m_original_rtn'
REQUEST_ID = '20261021_create_15m_original_train_03'
FEATURES = ['open', 'close', 'high', 'low', 'vwap', 'turn_fl']
FRAMES = ('fit', 'predict')


def read_snapshot(path: Path) -> DataBlock:
    # A snapshot is data, never executable instructions. Only mmap is accepted.
    if path.suffix != '.mmap' or not path.is_dir():
        raise ValueError(f'Expected an existing mmap snapshot: {path}')
    data = Load.mmap(path, values=True, index=True)
    return DataBlock(data['values'], data['index']['secid'], data['index']['date'], data['index']['feature'])


def audit(block: DataBlock, expected_dates=None):
    if block.empty or block.shape[2] != 16 or list(block.feature) != FEATURES:
        raise ValueError('15m_original: invalid/empty schema')
    if block.values.dtype != torch.float32:
        raise ValueError(f'15m_original: expected float32, got {block.values.dtype}')
    for name in ('secid', 'date'):
        index = getattr(block, name)
        if not np.array_equal(index, np.unique(index)):
            raise ValueError(f'15m_original: unsorted or duplicate {name}')
    if expected_dates is not None and not np.array_equal(block.date, expected_dates):
        raise ValueError('15m_original: incomplete date coverage')
    digest = hashlib.sha256()
    for index in (block.secid, block.date, block.feature):
        digest.update(json.dumps(index.tolist(), ensure_ascii=True).encode())
    finite = torch.zeros(6, dtype=torch.bool)
    daily = []
    usable_dates = []
    for start in range(0, len(block.date), 32):
        values = block.values[:, start:start + 32].detach().cpu().contiguous()
        digest.update(memoryview(values.numpy()))
        f = torch.isfinite(values)
        usable_dates.extend(int(date) for date, usable in
                            zip(block.date[start:start + 32], f.any(dim=(0, 2, 3)).tolist()) if usable)
        finite |= f.any(dim=(0, 1, 2))
        infs = torch.isinf(values).sum(dim=(0, 2)).tolist()
        nans = torch.isnan(values).sum(dim=(0, 2)).tolist()
        daily.extend({'date': int(d), 'inf': i, 'nan': n}
                     for d, i, n in zip(block.date[start:start + 32], infs, nans))
    if not finite.all():
        raise ValueError('15m_original: a feature has no finite values')
    internal_empty = np.setdiff1d(block.date[(block.date >= usable_dates[0]) &
                                            (block.date <= usable_dates[-1])], usable_dates)
    if len(internal_empty):
        raise ValueError(f'15m_original: internal completely unusable dates {internal_empty.tolist()}')
    return {'sha256': digest.hexdigest(), 'shape': list(block.shape), 'dtype': str(block.values.dtype),
            'features': FEATURES, 'daily': daily,
            'leading_empty_dates': block.date[block.date < usable_dates[0]].tolist(),
            'trailing_empty_dates': block.date[block.date > usable_dates[-1]].tolist()}


def find_snapshot(frame: str, manifest: str = ''):
    """Never infer pre-fix provenance from an old-looking filename alone."""
    paths = [Path(manifest)] if manifest else sorted((PATH.interim / 'autofix_backups').glob('*/report.json'))
    rejected = []
    for path in paths:
        doc = json.loads(path.read_text())
        if doc.get('baseline_commit') != BASELINE_COMMIT or doc.get('historical_snapshot') is not True:
            if manifest:
                raise ValueError('Snapshot manifest must attest historical_snapshot and exact baseline_commit')
            rejected.append(str(path))
            continue
        for item in doc.get('blocks', []):
            if item.get('key') != '15m' or item.get('frame') != frame:
                continue
            source = Path(item['backup'])
            if not source.is_absolute():
                source = path.parent / source
            block = read_snapshot(source)
            check = audit(block)
            if check['sha256'] != item.get('sha256'):
                raise ValueError(f'Snapshot digest mismatch: {source}')
            return block, {'kind': 'historical_snapshot', 'manifest': str(path),
                           'path': str(source), 'sha256': check['sha256']}
    return None, {'kind': 'legacy_formula_recomputed', 'unverified_manifests_skipped': rejected}


def input_eligibility(block: DataBlock):
    """Input-only 30-day eligibility; labels and period splits are checked by training.

    Chunk securities, not time: forward filling must preserve the full date history.
    """
    index = torch.arange(29, max(29, len(block.date)))
    counts = torch.zeros(len(index), dtype=torch.int64)
    for start in range(0, len(block.secid), 32):
        values = block.values[start:start + 32].clone().float()
        pre = torch.isfinite(values)
        values[~pre] = torch.nan
        DataBlock.autofill_values(values, block.feature, stretch=True)
        if len(index):
            # Exact fixed-window equivalent of DataOperator.finite_position for
            # 15m_original (30 days, step=1, no DivLast). Do not import the model
            # package here: its __init__ eagerly loads unrelated algorithm stacks.
            day_missing = (~pre).sum(dim=(2, 3))
            missing_ratio = day_missing.unfold(1, 30, 1).sum(-1).float() / float(30 * values.shape[2] * values.shape[3])
            day_unfilled = (~torch.isfinite(values)).sum(dim=(2, 3))
            completely_filled = day_unfilled.unfold(1, 30, 1).sum(-1) == 0
            counts += ((missing_ratio <= .1) & completely_filled).sum(dim=0).cpu()
    return {'kind': 'input_only_before_labels_and_period_split',
            'dates': block.date[29:].tolist(), 'counts': counts.tolist()}


def compare(current: DataBlock, original: DataBlock):
    if current.empty:
        return {'status': 'unavailable', 'reason': 'current 15m cache missing'}
    dates = np.intersect1d(current.date, original.date)
    secids = np.intersect1d(current.secid, original.secid)
    if not len(dates) or not len(secids):
        return {'status': 'unavailable', 'reason': 'no common coordinates'}
    rows = []
    for start in range(0, len(dates), 32):
        ds = dates[start:start + 32]
        a = current.align(secid=secids, date=ds, feature=FEATURES).values
        b = original.align(secid=secids, date=ds, feature=FEATURES).values
        changed = ~((a == b) | (torch.isnan(a) & torch.isnan(b)))
        for d, count in zip(ds, changed.sum(dim=(0, 2)).tolist()):
            rows.append({'date': int(d), 'changed_by_feature': count})
    return {'status': 'compared', 'common_securities': len(secids), 'daily': rows,
            'current_input_eligibility': input_eligibility(current)}


class SourceCoverageError(ValueError):
    """Keep structured preflight evidence without printing all valid dates."""
    def __init__(self, diagnostic: dict):
        self.diagnostic = diagnostic
        details = {key: diagnostic[key] for key in
                   ('frame', 'target_start', 'target_end', 'start', 'end', 'source_ranges', 'missing_sources')}
        super().__init__(f'{KEY}/{diagnostic["frame"]}: source coverage incomplete: {json.dumps(details)}')


def prepare(snapshot_manifest: str = ''):
    cutoff = int(CALENDAR.updated())
    available = {key: np.asarray(sorted(set(DB.dates('trade_ts', key, use_alt=True))))
                 for key in ('min', 'day')}
    jobs = []
    for frame in FRAMES:
        proc = PrePro_15m_original(frame, vb_level='never')
        proc.load_end = cutoff
        snapshot, provenance = find_snapshot(frame, snapshot_manifest)
        proc.snapshot = snapshot
        target_start = int(proc.load_start)
        dates = CALENDAR.range(target_start, cutoff)
        covered = np.array([], dtype=np.int64) if snapshot is None else snapshot.date
        usable = np.intersect1d(dates, np.union1d(covered,
                                               np.intersect1d(available['min'], available['day'])))
        actual = CALENDAR.range(int(usable[0]), int(usable[-1])) if len(usable) else np.array([], dtype=np.int64)
        required = np.setdiff1d(actual, covered)
        missing = {key: np.setdiff1d(required, ds).tolist() for key, ds in available.items()}
        item = {'key': KEY, 'frame': frame, 'status': 'preflight_passed',
                'target_start': target_start, 'target_end': cutoff,
                'start': int(usable[0]) if len(usable) else None,
                'end': int(usable[-1]) if len(usable) else None, 'provenance': provenance,
                'snapshot_dates': np.intersect1d(actual, covered).tolist(),
                'recomputed_dates': required.tolist(),
                'leading_unavailable_dates': dates[dates < usable[0]].tolist() if len(usable) else dates.tolist(),
                'trailing_unavailable_dates': dates[dates > usable[-1]].tolist() if len(usable) else [],
                'missing_sources': missing,
                'source_ranges': {key: [int(ds[0]), int(ds[-1])] if len(ds) else None
                                  for key, ds in available.items()},
                'path': str(DataBlock.path_preprocess(KEY, frame))}
        if not len(usable) or any(missing.values()):
            item['status'] = 'preflight_failed'
            raise SourceCoverageError(item)
        proc.load_start, proc.load_end = int(usable[0]), int(usable[-1])
        jobs.append((proc, item))
    return jobs


def request_training():
    from src.api.calls.worklist_state import WorklistState
    from src.api.task_monitor.scheduling.maintenance_gate import sandbox_request
    context = sandbox_request()
    return WorklistState(SCHEDULE).request_training(context['id'] if context is not None else REQUEST_ID, allow_unlisted=True)


def run_task(dry_run: bool = False, snapshot_manifest: str = ''):
    if dry_run:
        jobs = prepare(snapshot_manifest)
        result = [item for _, item in jobs]
        Logger.note(json.dumps(result))
        return result
    folder = PATH.interim / 'autofix_backups' / (
        '15m_original_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    report_path = folder / 'report.json'
    report = {'baseline_commit': BASELINE_COMMIT, 'historical_snapshot': False,
              'status': 'building', 'schedule': SCHEDULE,
              'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                              'numpy': np.__version__, 'pandas': pd.__version__},
              'known_differences': ['Global2Top retry correction retained',
                  'Current minute/day/label sources and dependencies may differ from historical snapshot',
                  'Input eligibility is not label-filtered training sample count'],
              'blocks': []}
    report['implementation_commit'] = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=PATH.main, text=True).strip()

    def save_report():
        temporary = report_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False))
        temporary.replace(report_path)

    save_report()
    Logger.note(f'15m_original report: {report_path}')
    try:
        jobs = prepare(snapshot_manifest)
        report['blocks'] = [item for _, item in jobs]
        save_report()
        # Back up both experiment targets before replacing either. Production 15m is read only.
        for proc, item in jobs:
            path = DataBlock.path_preprocess(KEY, proc.frame)
            if path.exists():
                backup = folder / proc.frame / path.name
                backup.parent.mkdir(parents=True, exist_ok=True)
                if path.is_dir():
                    shutil.copytree(path, backup)
                else:
                    shutil.copy2(path, backup)
                item['backup'] = str(backup)
        save_report()
        for proc, item in jobs:
            item['minute_sources'] = []
            proc.source_observer = item['minute_sources'].append
            block = proc.load_with_extension(reconstruct=True)
            expected = CALENDAR.range(proc.load_start, proc.load_end)
            # Validate the returned axis before any alignment can hide a missing date.
            check = audit(block, expected)
            item['leading_empty_dates'] = check['leading_empty_dates']
            item['trailing_empty_dates'] = check['trailing_empty_dates']
            start = len(check['leading_empty_dates'])
            end = len(block.date) - len(check['trailing_empty_dates'])
            block = block.slice_date(int(block.date[start]), int(block.date[end - 1]))
            expected = block.date.copy()
            item['saved_start'], item['saved_end'] = int(expected[0]), int(expected[-1])
            check = audit(block, expected)
            item['audit'] = check
            item['input_eligibility'] = input_eligibility(block)
            if proc.frame == 'fit' and not any(item['input_eligibility']['counts']):
                raise ValueError('15m_original: no usable 30-day input windows')
            current = DataBlock.load_dump(category='preprocess', preprocess_key='15m', frame=proc.frame)
            item['comparison'] = compare(current, block)
            del current
            proc.save_dump(block)
            del block
            saved = proc.load_dump()
            if audit(saved, expected) != check:
                raise ValueError(f'{KEY}/{proc.frame}: saved content differs')
            del saved
            item['status'] = 'verified'
            save_report()
        report['status'] = 'verified'
        save_report()
        report['training_request'] = request_training()
        report['status'] = 'training_requested'
    except Exception as exc:
        if isinstance(exc, SourceCoverageError):
            report['blocks'] = [exc.diagnostic]
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save_report()
    return report


@ScriptTool('autofix_create_15m_original')
def main(dry_run: bool = False, snapshot_manifest: str = '', **kwargs):
    run_task(dry_run=dry_run, snapshot_manifest=snapshot_manifest)


if __name__ == '__main__':
    main()
