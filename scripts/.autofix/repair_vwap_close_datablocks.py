# description: Rebuild intraday and weekly DataBlocks with infinite VWAP replaced by same-bar close
# content: Preflight sources, back up all targets, rebuild both frames and fit norms, then verify saved data.
# autofix: true
# email: true
# mode: shell
# parameters:
#   dry_run:
#     type: bool
#     default: false
#     desc: Check source coverage and print all targets without writing anything

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import torch

from src.proj import CALENDAR, DB, PATH, Logger
from src.proj.util.script import ScriptTool
from src.data import DataBlock
from src.data.preprocess import PrePros
from src.data.preprocess.hist_norm import HistNormTasks, PreProHistNorm

SOURCES = {'15m': ('15min', 'day'), '30m': ('30min', 'day'),
           '60m': ('60min', 'day'), '30mcont': ('30min', 'day'), 'week': ('day',)}
BARS = {'15m': 16, '30m': 8, '60m': 4, '30mcont': 8, 'week': 5}
FEATURES = ['open', 'close', 'high', 'low', 'vwap', 'turn_fl']
FRAMES = ('fit', 'predict')


def check_sources(proc, item, available):
    """Allow unavailable leading fit history only; include week's four warmup days."""
    key = item['key']
    warmup = 4 if key == 'week' else 0
    start, end = int(proc.load_start), int(proc.load_end)
    source_dates = {source: available[source][available[source] <= end] for source in SOURCES[key]}
    if any(not len(dates) for dates in source_dates.values()):
        raise ValueError(f'{key}/{proc.frame}: missing raw source dates')
    earliest = max(int(dates[0]) for dates in source_dates.values())
    effective = max(start, int(CALENDAR.td(earliest, warmup).td)) if proc.frame == 'fit' else start
    required_start = int(CALENDAR.td(effective, -warmup).td)
    expected = CALENDAR.range(required_start, end)
    missing = {source: np.setdiff1d(expected, dates).tolist() for source, dates in source_dates.items()}
    requested = CALENDAR.range(start, end)
    item['source_coverage'] = {
        'requested_start': start, 'effective_start': effective, 'end': end,
        'warmup_start': required_start, 'warmup_days': warmup,
        'omitted_leading_dates': requested[requested < effective].tolist(),
        'missing_dates': missing, 'use_alt': True,
    }
    if not len(expected) or effective > end or any(missing.values()):
        raise ValueError(f'{key}/{proc.frame}: incomplete source coverage: '
                         f'{ {source: dates[:20] for source, dates in missing.items()} }')
    proc.load_start = effective
    Logger.note(f'{key}/{proc.frame}: rebuild {effective}..{end}; raw warmup starts {required_start}')


def validate_block(block, key, expected_dates):
    """Bounded-memory validation and content digest; never sanitize other features."""
    if block.empty or block.shape[2] != BARS[key] or list(block.feature) != FEATURES:
        raise ValueError(f'{key}: empty or invalid schema: {block}')
    if len(np.unique(block.secid)) != len(block.secid) or not np.array_equal(block.date, expected_dates):
        raise ValueError(f'{key}: duplicate securities or rebuilt dates do not cover the requested window')
    finite = torch.zeros(len(FEATURES), dtype=torch.bool)
    infs = torch.zeros(len(FEATURES), dtype=torch.int64)
    digest = hashlib.sha256()
    for start in range(0, len(block.date), 64):
        values = block.values[:, start:start + 64].detach().cpu().contiguous()
        finite_values = torch.isfinite(values)
        usable_dates = finite_values.any(dim=(0, 2, 3))
        if not usable_dates.all():
            missing = block.date[start:start + 64][~usable_dates.numpy()].tolist()
            raise ValueError(f'{key}: dates contain no finite data: {missing}')
        finite |= finite_values.any(dim=(0, 1, 2))
        infs += torch.isinf(values).sum(dim=(0, 1, 2))
        digest.update(memoryview(values.numpy()))
    if infs.any():
        counts = {name: int(count) for name, count in zip(FEATURES, infs) if count}
        raise ValueError(f'{key}: residual Inf {counts}; only VWAP is repaired, other fields are not changed')
    if not finite.all():
        raise ValueError(f'{key}: all-NaN features {np.asarray(FEATURES)[~finite.numpy()].tolist()}')
    return {'shape': list(block.shape), 'dtype': str(block.values.dtype), 'sha256': digest.hexdigest()}


def backup_path(path, target):
    if not path.exists():
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    if path.is_dir():
        shutil.copytree(path, target)
    else:
        shutil.copy2(path, target)
    return str(target)


def run_task(dry_run: bool = False):
    cutoff = int(CALENDAR.updated())
    available = {source: np.asarray(sorted(set(DB.dates('trade_ts', source, use_alt=True))), dtype=np.int64)
                 for source in sorted({source for sources in SOURCES.values() for source in sources})}
    jobs, items = [], []
    for frame in FRAMES:
        for key in SOURCES:
            proc = PrePros.get_processor(key, frame, vb_level='never')
            proc.load_end = cutoff
            proc.DateChunkYears = 1
            item = {'key': key, 'frame': frame, 'status': 'pending',
                    'path': str(DataBlock.path_preprocess(key, frame)), 'vwap_repairs': []}
            jobs.append((proc, item))
            items.append(item)
    if dry_run:
        for proc, item in jobs:
            check_sources(proc, item, available)
        return items

    folder = PATH.interim / 'autofix_backups' / (
        'vwap_close_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    report_path = folder / 'report.json'
    report = {'cutoff': cutoff, 'status': 'checking_sources', 'blocks': items}

    def save_report():
        temporary = report_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False))
        temporary.replace(report_path)

    save_report()
    Logger.note(f'VWAP repair backup/report: {report_path}')
    try:
        # Check all sources, then back up all ten dumps and applicable norms,
        # before the first overwrite. An error cannot silently shorten history.
        for proc, item in jobs:
            check_sources(proc, item, available)
            item['status'] = 'sources_verified'
            save_report()
        for proc, item in jobs:
            key, frame = item['key'], item['frame']
            source = DataBlock.path_preprocess(key, frame)
            item['backup'] = backup_path(source, folder / key / frame / source.name)
            if frame == 'fit' and key in HistNormTasks:
                norm = PreProHistNorm.norm_path(key)
                item['norm_path'] = str(norm)
                item['norm_backup'] = backup_path(norm, folder / key / 'norm' / norm.name)
            item['status'] = 'backed_up'
            save_report()
        for proc, item in jobs:
            key, frame = item['key'], item['frame']
            item['status'] = 'rebuilding'
            save_report()

            def observe(record, item=item):
                # Source counts may overlap at yearly week-window boundaries.
                item['vwap_repairs'].append(record)
                Logger.note(f'{item["key"]}/{item["frame"]} VWAP fallback: {record}')
                save_report()

            proc.vwap_inf_repair_observer = observe
            block = proc.load_with_extension(reconstruct=True)
            dates = CALENDAR.range(proc.load_start, cutoff)
            check = validate_block(block, key, dates)
            secids = block.secid.copy()
            if frame == 'fit' and key in HistNormTasks:
                item['status'] = 'rebuilding_norm'
                save_report()
                proc.save_norm(block)
                norm = PreProHistNorm.load_keys([key]).get(key)
                if norm is None or not torch.isfinite(norm.avg).all() or not torch.isfinite(norm.std).all():
                    raise ValueError(f'{key}: missing or non-finite rebuilt historical norm')
                item['norm_verified'] = True
                del norm
            item['status'] = 'saving'
            save_report()
            proc.save_dump(block)
            del block
            saved = proc.load_dump()
            if validate_block(saved, key, dates) != check or not np.array_equal(saved.secid, secids):
                raise ValueError(f'{key}/{frame}: saved values or coordinates differ from rebuilt data')
            del saved
            item.update(status='verified', **check, start=int(dates[0]), end=int(dates[-1]))
            save_report()
        report['status'] = 'success'
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save_report()
    return items


@ScriptTool('autofix_repair_vwap_close_datablocks')
def main(dry_run: bool = False, **kwargs):
    run_task(dry_run=dry_run)


if __name__ == '__main__':
    main()
