"""Registration at an installation boundary, under the short admission lock."""
from __future__ import annotations

from datetime import datetime
import json
from zoneinfo import ZoneInfo

from .autofix_store import Store, state_dir
from .config import schedule_slots


def freeze(store, key, task, config):
    with store.connect() as db:
        db.execute('INSERT OR IGNORE INTO metadata VALUES (?,?)',
                   ('schedule-config:' + key, json.dumps({'task': task, 'config': config})))


def queued_config(store, key, current):
    with store.connect(readonly=True) as db:
        row = db.execute('SELECT value FROM metadata WHERE key=?', ('schedule-config:' + key,)).fetchone()
    return json.loads(row[0])['config'] if row else current


def cancel_disabled(config):
    if not state_dir().exists():
        return
    store = Store(state_dir())
    from .calendar_lanes import observation
    for key, task in store.scheduled():
        if not config['tasks'].get(task, {}).get('enabled'):
            store.schedule_state(key, 'cancelled')
            observation(store, key, status='cancelled', reason='task disabled by configuration publication')


def catch_up(directory, config, units, start, end, cron=''):
    """Only elapsed managed calendar triggers; never backfill earlier schedule slots."""
    if not config or config.get('scheduler_version', 1) < 3 or not state_dir().exists():
        return
    store = Store(state_dir())
    from .installer import cron_coverage, BEGIN, END
    from src.automation.paths import ROOT
    coverage, _, _ = cron_coverage(cron, config, ROOT, config['timezone'])
    managed_cron = cron.split(BEGIN, 1)[1].split(END, 1)[0] if BEGIN in cron else ''
    from .runtime import RunStore, members
    runs = RunStore(directory / 'runs.sqlite').active()
    from .calendar_lanes import observation
    for slot in range(int(start // 60) + 1, int(end // 60) + 1):
        local = datetime.fromtimestamp(slot * 60, ZoneInfo(config['timezone']))
        point = ((local.weekday() + 1) % 7, local.hour * 60 + local.minute)
        for schedule in config['schedules'].values():
            task = schedule['task']
            if (not config['tasks'][task]['enabled'] or
                    (f'learndl-schedule-{task}.timer' not in units and f'calendar {task} ' not in managed_cron) or
                    point not in schedule_slots(schedule) or point in coverage.get(task, set())):
                continue
            task_config = config['tasks'][task]
            request_slot = int(local.replace(hour=21, minute=1, second=0).timestamp() // 60) if task_config.get('window') == 'rcquant_evening' else slot
            key = f'{task}:{request_slot}'
            store.enqueue(task, request_slot)
            freeze(store, key, task, config)
            if (key, task) not in store.scheduled():
                continue
            busy = next((run for run in runs if run.get('lock_group') == task_config['lock_group'] and
                         (run['phase'] == 'starting' or members(run))), None)
            if busy:
                store.schedule_state(key, 'skipped_overlap')
                observation(store, key, status='skipped_overlap', blocked_by=busy['id'])
            else:
                for older, name in store.scheduled():
                    if name == task and int(older.rsplit(':', 1)[1]) < request_slot:
                        store.schedule_state(older, 'superseded')
                        observation(store, older, status='superseded', replaced_by=key)
