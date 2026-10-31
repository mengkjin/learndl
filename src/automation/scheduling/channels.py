"""Version 3 admission: independent lanes and bounded maintenance reservations.

Reservation transitions are serialized separately from short admission transactions.
Physical flock leases, not elapsed time or PIDs alone, protect running work.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
import json
import time

import portalocker

from .config import installed_config

KEY = 'channel_maintenance'
WAIT_SECONDS = 3600
COOLDOWN_SECONDS = 900
GROUPS = ('data', 'research', 'independent', 'manual', 'system', 'autofix')
BLOCKING = {'none': (), 'research': ('research',), 'research_and_data': ('research', 'data')}


def enabled():
    return (installed_config() or {}).get('scheduler_version', 1) >= 3


def initialize(store):
    with store.connect() as db:
        db.execute('INSERT OR IGNORE INTO metadata VALUES (?,?)', (KEY, json.dumps({'phase': 'idle'})))


def read(store):
    with store.connect(readonly=True) as db:
        row = db.execute('SELECT value FROM metadata WHERE key=?', (KEY,)).fetchone()
    if row is None:
        raise RuntimeError('Channel state missing; run scheduling installer apply')
    value = json.loads(row[0])
    if value.get('phase') not in {'idle', 'waiting', 'running', 'cooldown'}:
        raise RuntimeError('Invalid channel maintenance state')
    return value


def write(store, state):
    with store.connect() as db:
        db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)', (KEY, json.dumps(state)))


def _expire(store, state, now):
    if state['phase'] == 'waiting' and now >= state['deadline']:
        state = dict(state, phase='cooldown', cooldown_until=now + COOLDOWN_SECONDS,
                     reason='Maintenance admission timed out after one hour', expired_at=now)
        write(store, state)
        # Durable, separate from the execution ledger. SMTP remains auxiliary work.
        from .auxiliary import control_error
        control_error(RuntimeError(f"{state['owner']}: {state['reason']} (request {state['requested_at']})"))
    if state['phase'] == 'cooldown' and now >= state['cooldown_until']:
        state = {'phase': 'idle'}
        write(store, state)
    return state


def reserve(store, owner, groups, now=None):
    now = time.time() if now is None else now
    with portalocker.Lock(str(store.directory / 'reservation.lock'), timeout=0):
        state = _expire(store, read(store), now)
        if owner == 'git' and state['phase'] == 'waiting' and state.get('owner') != 'git':
            state = dict(state, owner=owner, groups=list(groups), preempted_owner=state['owner'])
            write(store, state)
        if state['phase'] == 'idle':
            state = dict(phase='waiting', owner=owner, groups=list(groups),
                         requested_at=now, deadline=now + WAIT_SECONDS)
            write(store, state)
        return state.get('owner') == owner and state['phase'] in {'waiting', 'running'}


def release(store, owner):
    with portalocker.Lock(str(store.directory / 'reservation.lock'), timeout=0):
        state = read(store)
        if state.get('owner') == owner and state['phase'] != 'cooldown':
            write(store, {'phase': 'idle'})


def running(store, owner):
    with portalocker.Lock(str(store.directory / 'reservation.lock'), timeout=0):
        state = _expire(store, read(store), time.time())
        if state.get('owner') != owner or state['phase'] not in {'waiting', 'running'}:
            raise RuntimeError('Maintenance reservation expired before execution')
        write(store, dict(state, phase='running'))


def blocker(store, task_class):
    from . import priority
    git = priority.read(store)
    if git['phase'] in {'checkout', 'blocked'} and task_class not in {'git', 'installation'}:
        return 'Git checkout requires verification'
    with portalocker.Lock(str(store.directory / 'reservation.lock'), timeout=0):
        state = _expire(store, read(store), time.time())
    if state['phase'] not in {'waiting', 'running'}:
        return None
    owner = 'git' if task_class == 'git' else task_class
    if owner == state['owner'] or task_class == 'installation':
        return None
    group = group_for(task_class)
    groups = state['groups']
    if isinstance(group, str) and isinstance(groups, list) and group in groups:
        return f"{state['owner']} reserves {group} until {state['deadline']}"
    return None


def group_for(task_class):
    if task_class.startswith('autofix:'):
        return 'autofix'
    return {'training': 'research', 'calendar': 'data'}.get(task_class, task_class)


@contextmanager
def leases(store, task_class):
    """Called under admission.lock; leases live for the entire execution scope."""
    group = group_for(task_class)
    groups = [group] if group in GROUPS else []
    exclusive = False
    if task_class.startswith('autofix:'):
        state = read(store)
        if state.get('owner') == task_class and state['phase'] in {'waiting', 'running'}:
            groups = sorted(set(groups + state['groups']))
            exclusive = True
    with ExitStack() as stack:
        descriptors = []
        for name in groups:
            stream = stack.enter_context((store.directory / f'channel-{name}.lock').open('a+'))
            portalocker.lock(stream, (portalocker.LOCK_EX if exclusive else portalocker.LOCK_SH) | portalocker.LOCK_NB)
            import os
            os.set_inheritable(stream.fileno(), True)
            descriptors.append(stream.fileno())
        yield tuple(descriptors)
