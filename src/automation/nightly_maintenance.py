"""Run resumable maintenance handlers under one stable schedule."""
from contextlib import closing
from datetime import datetime
from hashlib import sha256
import os
import sqlite3
import time
from zoneinfo import ZoneInfo

import polars as pl
import portalocker
import yaml

from src.automation.scheduling.nightly_window import load_window, manual_nightly_slot
from src.proj import Logger, MACHINE, PATH

HISTORY_COMPLETE = 'history complete'
HISTORY_NOTICE_DEFERRED = 'history notice deferred'


def rcquant_minutes(guard, slot: float) -> tuple[str, str]:
    from src.data.download.other_source.rcquant import bar_min
    from src.data.download.other_source.rcquant.initializer import QuotaExceeded
    from src.data.download.other_source.rcquant.sec_none_migration import EveningEnded, Migration, REPAIR
    from src.data.download.other_source.rcquant.sec_native_revision import DATE, REVISION, repair_native_day
    from src.data.download.other_source.rcquant.sec_storage_recovery import recover_storage, transcode_raw
    from src.data.download.other_source.rcquant.storage import GIB, SpaceDeferred, disk_full

    reserve = yaml.safe_load((PATH.main / 'configs/automation/nightly.yaml').read_text()).get('disk_reserve_gib', 50)
    if type(reserve) is not int or reserve < 0:
        raise ValueError('nightly disk_reserve_gib must be a nonnegative integer')

    root = bar_min.RQ_PATH
    state = root / '.sec_none_migration'
    state.mkdir(parents=True, exist_ok=True)
    night = datetime.fromtimestamp(slot, ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d')
    with portalocker.Lock(state / 'migration.lock', timeout=0), closing(sqlite3.connect(state / 'progress.sqlite')) as db:
        job = Migration(root, PATH.miscel / 'exfactor', db, started_at=None)
        job.guard = guard
        job.reserve_bytes = reserve * GIB
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
        reason = 'staging cleanup and secmin compression'
        try:
            recover_storage(job)
            reason = f'{DATE} native revision blocks downstream'
            if job.read('quota_exhausted', night):
                return 'quota deferred', reason
            guard()
            # Existing native receipts only need reads. If more work is required,
            # the native repair checks space before downloading and writing.
            if not repair_native_day(job):
                return 'waiting on data', reason
            source_receipt = job.read(f'{REVISION}_done', DATE)
            if source_receipt is not None:
                saved = transcode_raw(job, DATE, source_receipt)
                if saved is not None:
                    Logger.stdout(f'RCQuant {DATE}: secmin compressed, {saved / GIB:.2f} GiB reclaimed')
            reason = 'historical minute publication blocks downstream'
            history_stage = f'{REVISION}_history'
            recorded = job.read(history_stage, 'dates')
            dates = recorded if recorded and all(job.minute_path(day).is_file() for day in recorded) else None
            if dates is None:
                dates = job.run(revised_dates=(DATE,), rebuild=False)
                if dates is None:
                    return 'waiting on data', f'{reason}, history {job.history_progress}'
                job.mark(history_stage, 'dates', dates)
            job.history_progress = (len(dates), len(dates))
            if job.read(history_stage, 'downstream_handoff') != dates:
                from src.proj.util.web.emailer import Email

                reason = 'historical minute completion email'
                message = (
                    f'已完成 {len(dates)} 个交易日的 RCQuant 历史分钟迁移'
                    f'（{dates[0]} 至 {dates[-1]}）。本轮在下游重算前退出。'
                    '下一次手动或定时运行会继续重算特征、因子和 DataBlock。'
                )
                digest = sha256(','.join(map(str, dates)).encode()).hexdigest()
                if not Email.enqueue_once('RCQuant 历史分钟迁移完成', message,
                                          delivery_id=f'nightly:{REPAIR}:{REVISION}:{digest}'):
                    return HISTORY_NOTICE_DEFERRED, '邮件队列尚不可用，下次运行会先重试入队。'
                job.mark(history_stage, 'downstream_handoff', dates)
                return HISTORY_COMPLETE, message
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
        except SpaceDeferred as exc:
            return 'space deferred', f'{reason}, {exc}, history {job.history_progress}'
        except Exception as exc:
            if disk_full(exc):
                return 'space deferred', f'{reason}, {exc}, history {job.history_progress}'
            raise


HANDLERS = {'rcquant_minutes': rcquant_minutes}


def run():
    from src.data.download.other_source.rcquant.sec_none_migration import EveningEnded
    from src.data.download.other_source.rcquant.storage import disk_full

    if MACHINE.name != 'mengkjin-server':
        raise RuntimeError('Nightly maintenance is configured for mengkjin-server')
    slot_text = os.environ.get('LEARNDL_NIGHTLY_SLOT')
    slot = manual_nightly_slot(PATH.main, time.time()) if slot_text is None else float(slot_text)
    window = load_window(PATH.main, slot)
    cutoff = float(os.environ.get('LEARNDL_NIGHTLY_CUTOFF', window.cutoff))
    Logger.stdout(f'Nightly maintenance checkpoint cutoff {datetime.fromtimestamp(cutoff, ZoneInfo("Asia/Shanghai"))}',
                  vb_level=0)
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
            if not disk_full(exc):
                Logger.conclude(f'Nightly {name} failed: {exc}')
                raise
            status, details = 'space deferred', str(exc)
        summary = f'Nightly {name} {status}, {details}'
        Logger.conclude(summary)
        Logger.stdout(summary, vb_level=0)
        if status in (HISTORY_COMPLETE, HISTORY_NOTICE_DEFERRED, 'deadline deferred', 'space deferred'):
            break
