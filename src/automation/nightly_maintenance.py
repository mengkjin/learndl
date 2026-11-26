"""Run resumable maintenance handlers under one stable schedule."""
from contextlib import closing
from datetime import datetime, time as local_time, timedelta
import os
import sqlite3
import time
from zoneinfo import ZoneInfo

import polars as pl
import portalocker

from src.automation.scheduling.nightly_window import load_window
from src.proj import Logger, MACHINE, PATH


def rcquant_minutes(guard, slot: float) -> tuple[str, str]:
    from src.data.download.other_source.rcquant import bar_min
    from src.data.download.other_source.rcquant.initializer import QuotaExceeded
    from src.data.download.other_source.rcquant.sec_none_migration import EveningEnded, Migration, REPAIR
    from src.data.download.other_source.rcquant.sec_native_revision import DATE, REVISION, repair_native_day

    root = bar_min.RQ_PATH
    state = root / '.sec_none_migration'
    state.mkdir(parents=True, exist_ok=True)
    night = datetime.fromtimestamp(slot, ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d')
    with portalocker.Lock(state / 'migration.lock', timeout=0), closing(sqlite3.connect(state / 'progress.sqlite')) as db:
        job = Migration(root, PATH.miscel / 'exfactor', db, started_at=None)
        job.guard = guard
        Logger.stdout(f'Nightly RCQuant Polars {pl.__version__}, threads={pl.thread_pool_size()}')
        old = job.read('manifest', 'history') or {}
        published = sum(1 for _ in db.execute('SELECT item FROM checkpoints WHERE stage=?', (f'{REPAIR}_done',)))
        prepared = db.execute(
            'SELECT count(*) FROM checkpoints p WHERE p.stage=? AND NOT EXISTS '
            '(SELECT 1 FROM checkpoints d WHERE d.stage=? AND d.item=p.item)',
            (f'{REPAIR}_prepared', f'{REPAIR}_done'),
        ).fetchone()[0]
        total = len(old.get('dates', [])) + len(old.get('missing', [])) + len(old.get('mapping_errors', {}))
        job.history_progress = (published, total)
        Logger.stdout(f'Nightly RCQuant reused ledger {state / "progress.sqlite"}, '
                      f'{published} published, {prepared} staged, {max(0, total-published)} pending')
        reason = f'{DATE} native revision blocks downstream'
        try:
            if job.read('quota_exhausted', night):
                return 'quota deferred', reason
            guard()
            if not repair_native_day(job):
                return 'waiting on data', reason
            reason = 'historical minute publication blocks downstream'
            recorded = job.read(f'{REVISION}_history', 'dates')
            dates = recorded if recorded and all(job.minute_path(day).is_file() for day in recorded) else None
            if dates is None:
                dates = job.run(revised_dates=(DATE,), rebuild=False)
                if dates is None:
                    return 'waiting on data', f'{reason}, history {job.history_progress}'
                job.mark(f'{REVISION}_history', 'dates', dates)
            job.history_progress = (len(dates), len(dates))
            guard()
            from src.data.download.other_source.rcquant.sec_minute_downstream import rebuild_minute_consumers
            reason = f'downstream replay, native audit {state / REVISION}'
            if not rebuild_minute_consumers(job, dates, job.require_record(f'{REVISION}_done', DATE)):
                return 'waiting on data', reason
            return 'complete', f'{len(dates)} history dates and all downstream stages complete'
        except QuotaExceeded:
            job.mark('quota_exhausted', night)
            return 'quota deferred', f'{reason}, history {job.history_progress}'
        except EveningEnded:
            return 'deadline deferred', f'{reason}, history {job.history_progress}'


HANDLERS = {'rcquant_minutes': rcquant_minutes}


def run():
    from src.data.download.other_source.rcquant.sec_none_migration import EveningEnded

    if MACHINE.name != 'mengkjin-server':
        raise RuntimeError('Nightly maintenance is configured for mengkjin-server')
    slot_text = os.environ.get('LEARNDL_NIGHTLY_SLOT')
    if slot_text is None:
        now = datetime.now(ZoneInfo('Asia/Shanghai'))
        evening = datetime.combine(now.date(), local_time(21), now.tzinfo)
        # A manual continuation after midnight belongs to the previous night.
        if now < evening:
            evening -= timedelta(days=1)
        slot = evening.timestamp()
    else:
        slot = float(slot_text)
    window = load_window(PATH.main, slot)
    cutoff = float(os.environ.get('LEARNDL_NIGHTLY_CUTOFF', window.cutoff))
    if set(window.tasks) - HANDLERS.keys() or len(set(window.tasks)) != len(window.tasks):
        raise ValueError('Unknown or duplicate nightly maintenance tasks')

    def guard():
        if time.time() >= cutoff:
            raise EveningEnded

    for name in window.tasks:
        try:
            guard()
            status, details = HANDLERS[name](guard, slot)
        except EveningEnded:
            status, details = 'deadline deferred', 'before starting the next task'
        except Exception as exc:
            Logger.conclude(f'Nightly {name} failed: {exc}')
            raise
        summary = f'Nightly {name} {status}, {details}'
        Logger.conclude(summary)
        Logger.stdout(summary, vb_level=0)
        if status == 'deadline deferred':
            break
