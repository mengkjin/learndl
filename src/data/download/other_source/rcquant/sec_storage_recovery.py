"""Reclaim staged duplicates and transcode published stock minutes without repairing them again."""
from __future__ import annotations

__all__ = [
    'TRANSCODE',
    'RECOVERY_RESERVE',
    'published_matches',
    'clean_staged',
    'transcode_raw',
    'recover_storage',
]

from itertools import groupby
import json
import os
from pathlib import Path
from time import perf_counter

import pandas as pd
import polars as pl
from polars.testing import assert_frame_equal

from src.proj import Logger

from . import bar_min
from .sec_native_revision import DATE, REVISION
from .sec_none_migration import REPAIR, SEC, file_matches, fingerprint, sync_directory, write_frame
from .storage import GIB, RAW_IPC, check_space, date_lock, frame_bytes

TRANSCODE = 'secmin_zstd_v1'
RECOVERY_RESERVE = 2 * GIB


def published_matches(job, date, output):
    target = Path(output['target'])
    if target == job.minute_path(date):
        conversion = job.read(TRANSCODE, date)
        if conversion and conversion['before'] == output['digest']:
            if file_matches(target, conversion['after']):
                return True
    return file_matches(target, output['digest'])


def _allocated(path):
    stat = path.stat()
    return stat.st_blocks * 512 if stat.st_nlink == 1 else 0


def clean_staged(job, revisions):
    files = retained = reclaimed = 0
    for revision, date, prepared in revisions:
        job.guard()
        for output in prepared['outputs']:
            target, pending = Path(output['target']), Path(output['staged'])
            if not pending.is_file() or pending.parent != target.parent / f'.{revision}_stage':
                continue
            with date_lock(job.root, SEC, date):
                digest = output['digest']
                published = job.read(f'{revision}_published', f'{date}/{target.name}')
                # A completed date is also proof of publication for older ledgers.
                if published != digest and job.read(f'{revision}_done', date) is None:
                    retained += 1
                    continue
                if file_matches(target, digest) and file_matches(pending, digest):
                    reclaimed += _allocated(pending)
                    pending.unlink()
                    sync_directory(pending.parent)
                    files += 1
                else:
                    retained += 1
    Logger.stdout(f'Nightly RCQuant staging cleanup: {files} duplicates removed, '
                  f'{reclaimed / GIB:.2f} GiB reclaimed, {retained} retained')


def transcode_raw(job, date, prepared):
    """Return the bytes reclaimed, or None when no conversion was needed."""
    if prepared.get('preserve_raw'):
        return None
    target = job.minute_path(date)
    output = next((x for x in prepared['outputs'] if Path(x['target']) == target), None)
    if output is None or output.get('ipc_compression') == RAW_IPC:
        return None
    stage = target.with_name(f'.{target.name}.{TRANSCODE}.pending')
    with date_lock(job.root, SEC, date):
        receipt = job.read(TRANSCODE, date)
        if receipt and receipt['before'] != output['digest']:
            receipt = None
        if receipt and receipt.get('complete'):
            return None
        current = fingerprint(target) if target.is_file() else None
        if receipt and current == receipt['after']:
            _finish(job, date, target, receipt)
            return receipt['saved']
        if current != output['digest']:
            raise RuntimeError(f'{date}: published secmin checksum differs, cannot transcode {target}')
        reclaimable = _allocated(target)
        if not receipt or not file_matches(stage, receipt['after']):
            frame = pl.read_ipc(target, memory_map=False)
            reserve = RECOVERY_RESERVE if reclaimable > 0 else job.reserve_bytes
            check_space(target.parent, frame_bytes(frame), reserve, operation=f'{date} transcode secmin')
            write_frame(frame, stage, ipc_compression=RAW_IPC)
            assert_frame_equal(frame, pl.read_ipc(stage, memory_map=False), check_exact=True)
            receipt = {'before': current, 'after': fingerprint(stage),
                       'saved': reclaimable - _allocated(stage)}
            # An existing daily receipt is keyed to the physical file identity.
            daily_receipt = target.parent / '.none_published' / target.name
            if daily_receipt.is_file():
                previous = bar_min.minute_identity(target)
                if pd.read_feather(daily_receipt).to_dict('records') == [previous]:
                    receipt['daily_identity_before'] = previous
            job.mark(TRANSCODE, date, receipt)
        if receipt['saved'] <= 0:
            job.check_space(target.parent, 0, operation=f'{date} replace secmin without net savings')
        os.replace(stage, target)
        sync_directory(target.parent)
        _finish(job, date, target, receipt)
        return receipt['saved']


def _finish(job, date, target, receipt):
    if 'daily_identity_before' in receipt:
        daily_receipt = target.parent / '.none_published' / target.name
        current = pd.read_feather(daily_receipt).to_dict('records') if daily_receipt.is_file() else []
        if current == [receipt['daily_identity_before']]:
            write_frame(pd.DataFrame([bar_min.minute_identity(target)]), daily_receipt)
    job.mark(TRANSCODE, date, {**receipt, 'complete': True})


def recover_storage(job):
    records = []
    for revision in (REPAIR, REVISION):
        rows = job.db.execute('SELECT item, value FROM checkpoints WHERE stage=?',
                              (f'{revision}_prepared',)).fetchall()
        records.extend((revision, int(date), json.loads(value)) for date, value in rows)
    clean_staged(job, records)

    candidates = {}
    for revision, date, prepared in records:
        if revision == REPAIR and date == DATE and job.read(f'{REVISION}_prepared', date):
            continue
        if job.read(f'{revision}_done', date) is not None:
            candidates[date] = prepared
    # Size is used only to prioritize reclaiming bytes before more output is written.
    months = {}
    for month, dates in groupby(sorted(candidates), key=lambda day: day // 100):
        months[month] = list(dates)

    def size(month):
        total = 0
        for day in months[month]:
            receipt = job.read(TRANSCODE, day)
            target = job.minute_path(day)
            if not (receipt and receipt.get('complete')) and target.is_file():
                total += target.stat().st_size
        return total

    Logger.stdout(f'Nightly RCQuant secmin Zstd: checking {len(candidates)} published dates')
    completed = 0
    for month in sorted(months, key=size, reverse=True):
        converted = reused = reclaimed = 0
        start = perf_counter()
        try:
            for date in months[month]:
                job.guard()
                saved = transcode_raw(job, date, candidates[date])
                if saved is None:
                    reused += 1
                else:
                    converted += 1
                    reclaimed += saved
                completed += 1
        finally:
            if converted:
                Logger.stdout(f'Nightly RCQuant secmin Zstd {month}: {converted} converted, '
                              f'{reused} reused, {reclaimed / GIB:.2f} GiB reclaimed, '
                              f'{completed}/{len(candidates)} checked, {perf_counter()-start:.1f}s')
    Logger.stdout(f'Nightly RCQuant secmin Zstd: {completed} dates checked')
