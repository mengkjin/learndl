"""Short calendar registration, independent per-task workers and durable reasons."""
from __future__ import annotations

from datetime import datetime
import json
import os
import time
from zoneinfo import ZoneInfo

import portalocker

from .config import installed_config
from .autofix_store import Store, state_dir
from .runtime import RunStore, members


def observation(store, key, **fields):
    name = 'calendar:' + key
    with store.connect() as db:
        row = db.execute('SELECT value FROM metadata WHERE key=?', (name,)).fetchone()
        value = json.loads(row[0]) if row else {'request_id': key, 'scheduled_at': int(key.rsplit(':', 1)[1]) * 60}
        value.update(fields, observed_at=time.time())
        db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)', (name, json.dumps(value)))


def active(task):
    group = task['lock_group']
    return next((r for r in RunStore().active() if r.get('lock_group') == group and
                 (r['phase'] == 'starting' or members(r))), None)


def register(task_id):
    config = installed_config()
    task = config['tasks'][task_id]
    store = Store(state_dir())
    slot = int(os.environ.get('LEARNDL_SCHEDULE_SLOT', int(time.time() // 60)))
    if task.get('window') == 'rcquant_evening':
        planned = datetime.fromtimestamp(slot*60, ZoneInfo(config['timezone'])).replace(hour=21, minute=1, second=0, microsecond=0)
        slot = int(planned.timestamp() // 60)
    key = f'{task_id}:{slot}'
    with portalocker.Lock(str(store.directory / 'admission.lock'), mode='r+', timeout=10):
        store.enqueue(task_id, slot)
        if (key, task_id) not in store.scheduled():
            return 0
        current = active(task)
        observation(store, key, enqueued_at=time.time(), channel=task['channel'])
        if current:
            store.schedule_state(key, 'skipped_overlap')
            observation(store, key, status='skipped_overlap', blocked_by=current['id'])
        else:
            for older, name in store.scheduled():
                if name == task_id and older != key and int(older.rsplit(':', 1)[1]) < slot:
                    store.schedule_state(older, 'superseded')
                    observation(store, older, status='superseded', replaced_by=key)
    return 0


def eligibility(task_id, key, config, now=None):
    now = time.time() if now is None else now
    planned = int(key.rsplit(':', 1)[1]) * 60
    local = datetime.fromtimestamp(now, ZoneInfo(config['timezone']))
    scheduled = datetime.fromtimestamp(planned, ZoneInfo(config['timezone']))
    task = config['tasks'][task_id]
    if not task['enabled']:
        return 'expired', 'task disabled'
    if task.get('window') == 'rcquant_evening':
        if scheduled.date() != local.date():
            return 'expired', 'RCQuant request passed its calendar day'
        if (local.hour, local.minute) < (21, 1):
            return 'waiting', 'RCQuant evening window has not opened'
        if local.hour == 23 and local.minute >= 59:
            return 'expired', 'RCQuant download deadline reached'
        from src.data.download.other_source.rcquant.bar_min import daily_update_succeeded_today
        if not daily_update_succeeded_today():
            return 'waiting', 'daily_update has not succeeded today'
    return 'ready', ''


def worker(task_id):
    from .maintenance_gate import permit, MaintenanceBusy
    from .runner import _run
    config = installed_config()
    task = config['tasks'][task_id]
    store = Store(state_dir())
    try:
        with portalocker.Lock(str(store.directory / f'lane-{task_id}.lock'), timeout=0):
            for key, name in store.scheduled():
                if name != task_id:
                    continue
                status, reason = eligibility(task_id, key, config)
                if status != 'ready':
                    if status == 'expired':
                        store.schedule_state(key, status)
                    observation(store, key, status=status, reason=reason)
                    continue
                import psutil
                if psutil.virtual_memory().available / psutil.virtual_memory().total < .10:
                    observation(store, key, status='waiting', reason='less than 10% system memory available')
                    break
                current = active(task)
                if current:
                    store.schedule_state(key, 'skipped_overlap')
                    observation(store, key, status='skipped_overlap', blocked_by=current['id'])
                    continue
                try:
                    with permit(blocking=False, task_class=task['channel']):
                        # Resolve overlap atomically at the underlying script lock too.
                        store.schedule_state(key, 'starting')
                        observation(store, key, status='starting', started_at=time.time(),
                                    delay_seconds=time.time() - int(key.rsplit(':', 1)[1]) * 60)
                        try:
                            code = _run(task_id, request_id=key)
                        except BaseException:
                            store.schedule_state(key, 'unknown')
                            observation(store, key, status='unknown', reason='launcher outcome requires reconciliation')
                            raise
                        status = 'skipped_overlap' if code == 76 else 'finished' if code == 0 else 'failed'
                        store.schedule_state(key, status)
                        observation(store, key, status=status, returncode=code, finished_at=time.time())
                except MaintenanceBusy as exc:
                    observation(store, key, status='waiting', reason=str(exc))
                # At most one request per worker tick; the next tick re-reads state.
                break
    except portalocker.LockException:
        pass
    return 0
