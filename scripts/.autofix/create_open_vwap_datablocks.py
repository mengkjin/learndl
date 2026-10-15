# description: Build open and VWAP label DataBlocks (fit and predict)
# content: Recalculate only ret5/10/20_open/vwap, then rebuild y_open/y_vwap for both frames.
# autofix: true
# email: false
# mode: shell
# parameters:
#   dry_run:
#     type: bool
#     default: false
#     desc: Print the bounded plan without writing data or backups

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch

from src.proj import CALENDAR, DB, PATH, Logger
from src.proj.util.script import ScriptTool
from src.data import DataBlock
from src.data.preprocess import PrePros
from src.data.update.custom.labels import ClassicLabelsUpdater, calc_price_labels

SOURCES = ('y_open', 'y_vwap')
FRAMES = ('fit', 'predict')


def make_plan(end: int) -> list[dict]:
    """Only mature labels: T+1+n must not exceed the frozen data cutoff."""
    jobs = []
    for price in ('open', 'vwap'):
        for days in (5, 10, 20):
            last = CALENDAR.td(end, -days - 1).as_int()
            dates = CALENDAR.range(ClassicLabelsUpdater.START_DATE, last)
            if len(dates) == 0:
                raise ValueError(f'No mature dates for {price}/{days} at {end}')
            jobs.append({'key': f'ret{days}_{price}', 'price': price, 'days': days, 'dates': dates})
    return jobs


def backup_existing(paths: list[Path], folder: Path) -> list[dict]:
    """Copy every existing target before any data is overwritten."""
    records = []
    for i, path in enumerate(paths):
        target = folder / f'{i:02d}_{path.name}'
        if path.exists():
            if path.is_dir():
                shutil.copytree(path, target)
            else:
                shutil.copy2(path, target)
        records.append({'original': str(path), 'backup': str(target) if path.exists() else None})
    return records


def validate_labels(frame, field: str, context: str):
    if frame is None or frame.empty or not {'secid', field}.issubset(frame.columns):
        raise ValueError(f'{context}: missing label data or schema')
    if frame.secid.isna().any() or frame.secid.duplicated().any():
        raise ValueError(f'{context}: null or duplicate secid')
    values = frame[field].to_numpy(dtype=float)
    if np.isinf(values).any() or not np.isfinite(values).any():
        raise ValueError(f'{context}: no finite labels or infinite values')


def validate_block(block: DataBlock, source: str, frame: str):
    price = source.removeprefix('y_')
    expected = {f'{kind}_{days}_{price}' for kind in ('rtn', 'std') for days in (5, 10, 20)}
    if block.empty or block.shape[2] != 1 or set(block.feature) != expected or len(block.feature) != 6:
        raise ValueError(f'{source}/{frame}: empty or invalid DataBlock schema')
    if len(np.unique(block.secid)) != len(block.secid) or len(np.unique(block.date)) != len(block.date):
        raise ValueError(f'{source}/{frame}: duplicate coordinate keys')
    for index, feature in enumerate(block.feature):
        values = block.values[..., index]
        if torch.isinf(values).any() or not torch.isfinite(values).any():
            raise ValueError(f'{source}/{frame}/{feature}: no finite labels or infinite values')


def run_task(dry_run: bool = False):
    end = int(CALENDAR.updated())
    jobs = make_plan(end)
    summary = [{'key': job['key'], 'start': int(job['dates'][0]), 'end': int(job['dates'][-1]),
                'dates': len(job['dates'])} for job in jobs]
    Logger.note(f'New price label rebuild (T+1 to T+1+n), cutoff={end}: {summary}')
    Logger.note(f'DataBlocks: {[(source, frame) for frame in FRAMES for source in SOURCES]}')
    if dry_run:
        return summary

    backup = PATH.main / 'data' / 'Interim' / 'autofix_backups' / (
        'open_vwap_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid4().hex[:8])
    backup.mkdir(parents=True, exist_ok=False)
    report_path = backup / 'report.json'
    report = {'cutoff': end, 'labels': summary, 'status': 'backing_up', 'completed': []}
    report_path.write_text(json.dumps(report, indent=2))
    try:
        paths = [DB.DBPath('labels_ts', job['key']).parent for job in jobs]
        paths += [DataBlock.path_preprocess(source, frame) for frame in FRAMES for source in SOURCES]
        report['backups'] = backup_existing(paths, backup)
        report['status'] = 'running'
        report_path.write_text(json.dumps(report, indent=2))
        Logger.note(f'Backups and progress: {report_path}')
        for job in jobs:
            field = f"rtn_{job['days']}_{job['price']}"
            for date in job['dates']:
                date = int(date)
                label = calc_price_labels(date, job['days'], job['price'])
                validate_labels(label, field, f"{job['key']}/{date}")
                if not DB.save(label, 'labels_ts', job['key'], date, overwrite=True, vb_level='never'):
                    raise RuntimeError(f"Failed to save {job['key']}/{date}")
            stored = set(DB.dates('labels_ts', job['key']))
            if not set(map(int, job['dates'])).issubset(stored):
                raise RuntimeError(f"Incomplete saved dates for {job['key']}")
            report['completed'].append(job['key'])
            report_path.write_text(json.dumps(report, indent=2))
            Logger.note(f"Generated {job['key']}: {len(job['dates'])} dates")

        # All six raw sources must finish before the first DataBlock is touched.
        for frame in FRAMES:
            for source in SOURCES:
                proc = PrePros.get_processor(source, frame, vb_level='never')
                proc.load_end = end
                proc.DateChunkYears = 1
                block = proc.load_with_extension(reconstruct=True)
                validate_block(block, source, frame)
                proc.save_dump(block)
                del block
                saved = proc.load_dump()
                validate_block(saved, source, frame)
                del saved
                report['completed'].append(f'{source}/{frame}')
                report_path.write_text(json.dumps(report, indent=2))
        report['status'] = 'success'
    except Exception as exc:
        report['status'] = 'failed'
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2))
    Logger.note(f'Open/VWAP label DataBlocks complete. Backup/report: {report_path}')


@ScriptTool('autofix_create_open_vwap_datablocks')
def main(dry_run: bool = False, **kwargs):
    run_task(dry_run=dry_run)


if __name__ == '__main__':
    main()
