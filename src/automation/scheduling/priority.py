"""Durable admission policy, serialized by maintenance_gate's admission lock.

The installed generation opts in explicitly; updating Git never installs services.
No execution identities are created or reset by this module.
"""
from __future__ import annotations

import json
import math
import time

from .config import installed_config

KEY = 'scheduling_git'


def enabled():
    return (installed_config() or {}).get('scheduler_version', 1) >= 2


def read(store):
    with store.connect(readonly=True) as db:
        row = db.execute('SELECT value FROM metadata WHERE key=?', (KEY,)).fetchone()
    if row is None:
        raise RuntimeError('Scheduling priority state missing; restore it or run installer apply')
    value = json.loads(row[0])
    if (not isinstance(value, dict) or value.get('phase') not in {'idle', 'draining', 'checkout', 'blocked'}
            or type(value.get('last_completed')) not in (int, float)
            or not math.isfinite(value['last_completed']) or value['last_completed'] < 0
            or (value['phase'] in {'checkout', 'blocked'} and not value.get('target'))):
        raise RuntimeError('Invalid scheduling priority state')
    return value


def write(store, value):
    with store.connect() as db:
        db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)', (KEY, json.dumps(value)))


def initialize(store):
    with store.connect() as db:
        db.execute('INSERT OR IGNORE INTO metadata VALUES (?,?)',
                   (KEY, json.dumps({'phase': 'idle', 'last_completed': 0})))


def git_due(store, now=None):
    state = read(store)
    if state['phase'] in {'draining', 'checkout', 'blocked'}:
        return True
    options = (installed_config() or {}).get('watchdog', {}).get('jobs', {}).get('git_auto_update', {})
    return bool(options.get('enabled') and
                (time.time() if now is None else now) >= state['last_completed'] + options['interval_seconds'])


def blocker(store, task_class):
    """Called only after acquiring the physical gate, under admission.lock."""
    from . import channels
    if channels.enabled():
        return channels.blocker(store, task_class)
    if not enabled() or task_class in {'git', 'installation'}:
        return None
    if git_due(store):
        return 'git_auto_update is due or awaiting checkout verification'
    if task_class == 'calendar':
        return None
    if store.scheduled():
        return 'calendar tasks are queued'
    if task_class == 'autofix':
        return None
    if store.enabled():
        from .autofix import discover
        with store.connect(readonly=True) as db:
            from pathlib import Path
            root = Path(db.execute("SELECT value FROM metadata WHERE key='root'").fetchone()[0])
        if not discover(store, root):
            return 'autofix manifest is invalid; discovery paused'
        queued = [row['id'] for row in store.rows() if row['phase'] == 'queued']
        if queued:
            return f'autofix waiting: {queued[0]}'
    return None


def status():
    from .autofix_store import Store, state_dir
    from .config import runtime_dir
    if not enabled():
        return {'scheduler_version': 1, 'reason': 'Run installer plan/apply to enable priority workers'}
    store = Store(state_dir())
    helpers = {}
    from .auxiliary import NAMES
    for name in NAMES:
        path = runtime_dir() / 'auxiliary' / f'{name}.json'
        helpers[name] = json.loads(path.read_text()) if path.exists() else {'status': 'not observed'}
    from . import channels
    extra = {}
    if channels.enabled():
        with store.connect(readonly=True) as db:
            observations = [json.loads(row[0]) for row in db.execute("SELECT value FROM metadata WHERE key LIKE 'calendar:%' ORDER BY json_extract(value, '$.scheduled_at') DESC LIMIT 100")]
        extra = {'maintenance': channels.read(store), 'calendar_observations': observations}
    return {'scheduler_version': 3 if channels.enabled() else 2, **extra, 'git': read(store), 'git_due': git_due(store),
            'calendar_queue': store.scheduled(),
            'autofix_queue': [r['id'] for r in store.rows() if r['phase'] == 'queued'],
            'helpers': helpers}
