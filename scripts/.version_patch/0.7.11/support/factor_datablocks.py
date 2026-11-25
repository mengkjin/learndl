"""Rebuild the factor DataBlock caches affected by the loader regression."""

import errno
import json
import os
import shutil

import numpy as np

from src.automation.version_patch import load_patch_support
from src.data.preprocess.processors import factors
from src.data.util import DataBlock
from src.proj import CALENDAR, Logger, MACHINE, Save
from src.res.factor.calculator import FactorCalculator


LAST_GOOD_DATE = 20260410
REPORT = 'factor-datablocks.json'


def _backup_file(source, target):
    try:
        return os.link(source, target)
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        return shutil.copy2(source, target)


def _backup(ctx, path, frame):
    target = ctx.backup_dir / frame / path.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if path.is_dir():
        shutil.copytree(path, target, copy_function=_backup_file)
    else:
        _backup_file(path, target)
    Logger.stdout(f'Backed up {path} to {target}', vb_level=0)


def _source_window(processor, start):
    dates = CALENDAR.range(start, processor.load_end, 'td')
    available = np.zeros(len(dates), dtype=bool)
    if not len(dates):
        return dates
    for calc in FactorCalculator.iter(category0=processor.category0, category1=processor.category1):
        stored = calc.stored_dates(start, processor.load_end)
        if stored:
            available |= np.isin(dates, stored.dates)
    return dates[available]


def _has_values(block, date):
    return (not block.empty and date in block.date
            and bool(block.loc(date=[date]).isfinite().any()))


def _rebuild(ctx, processor, allow_create):
    key, frame = processor.key, processor.frame
    path = DataBlock.path_preprocess(key, frame, find_if_not_exists=False)
    # Recover if a previous attempt stopped between the two mmap publish renames.
    interrupted = path.with_name(f'{path.name}.old')
    if interrupted.exists() and not path.exists():
        interrupted.rename(path)
    original = DataBlock.path_preprocess(key, frame)
    if not original.exists() and not allow_create:
        return None

    old = processor.load_dump() if frame == 'fit' else DataBlock()
    start = processor.load_start
    if not old.empty and (last := old.last_meaningful_date) >= start:
        start = max(start, min(last, LAST_GOOD_DATE) + 1)
    available = _source_window(processor, start)
    if not len(available):
        Logger.stdout(f'{key}/{frame}: no stored factor dates from {start} through {processor.load_end}', vb_level=0)
        return None

    first, last = int(available[0]), int(available[-1])
    Logger.stdout(f'{key}/{frame}: rebuilding from {first} through {last}', vb_level=0)
    rebuilt = processor.pre_process(first, last)
    if not _has_values(rebuilt, last):
        raise ValueError(f'{key}/{frame}: no usable factor values on the latest stored date {last}')
    block = old.merge_others(rebuilt, inplace=True)
    if original.exists() and not (ctx.backup_dir / frame / original.name).exists():
        _backup(ctx, original, frame)
    # Publish through the atomic mmap writer. Existing legacy file dumps remain readable backups.
    Save.mmap(block.values, path, index={'date': block.date, 'secid': block.secid, 'feature': block.feature})
    return last


def apply(ctx):
    # A pending rebuild must repair its source before the later version can run.
    load_patch_support('0.7.13', 'turnover_snapshot').apply(ctx)
    results = {'fit': {}, 'predict': {}}
    for frame, allow_create in (('fit', MACHINE.cuda_server), ('predict', MACHINE.updatable)):
        for name in factors.__all__:
            processor = getattr(factors, name)(frame)
            latest = _rebuild(ctx, processor, allow_create)
            if latest is not None:
                results[frame][processor.key] = latest
    (ctx.backup_dir / REPORT).write_text(json.dumps(results, indent=2), encoding='utf-8')


def verify(ctx):
    results = json.loads((ctx.backup_dir / REPORT).read_text(encoding='utf-8'))
    for frame, blocks in results.items():
        for key, date in blocks.items():
            block = DataBlock.load_dump(category='preprocess', preprocess_key=key, frame=frame)
            if not _has_values(block, date):
                raise ValueError(f'{key}/{frame}: rebuilt dump has no usable factor values on {date}')
            del block
        Logger.stdout(f'{frame}: verified {len(blocks)} rebuilt factor DataBlocks', vb_level=0)
    return True
