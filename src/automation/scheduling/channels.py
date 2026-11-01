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


def request(store, owner):
    with store.connect(readonly=True) as db:
        row = db.execute('SELECT value FROM metadata WHERE key=?', (KEY + ':' + owner,)).fetchone()
    if row:
        return json.loads(row[0])
    # Migrate only the passive record. A live legacy reservation remains authoritative.
    legacy = read(store)
    return legacy if legacy.get('owner') == owner else {}


def save_request(store, state):
    with store.connect() as db:
        db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)',
                   (KEY + ':' + state['owner'], json.dumps(state)))


def _expire(store, state, now):
    if state.get('phase') == 'waiting' and now >= state['deadline']:
        state = dict(state, phase='cooldown', cooldown_until=now + COOLDOWN_SECONDS,
                     reason='Maintenance admission timed out after one hour', expired_at=now)
        if now - state.get('notified_at', 0) >= 86400:
            from .auxiliary import control_error
            control_error(RuntimeError(f"{state['owner']}: {state['reason']} (request {state['requested_at']})"))
            state['notified_at'] = now
    if state.get('phase') == 'cooldown' and now >= state['cooldown_until']:
        state = dict(state, phase='waiting', deadline=now + WAIT_SECONDS)
    return state


def reserve(store, owner, groups, now=None):
    """Record an independent waiting request, without reserving business admission."""
    now = time.time() if now is None else now
    with portalocker.Lock(str(store.directory / 'reservation.lock'), timeout=0):
        active = read(store)
        state = request(store, owner)
        if not state:
            state = dict(phase='waiting', owner=owner, groups=list(groups),
                         requested_at=now, deadline=now + WAIT_SECONDS)
        state = _expire(store, state, now)
        save_request(store, state)
        if active['phase'] != 'running':
            write(store, state)  # Compatibility/status projection, not an admission barrier.
        return (state['phase'] in {'waiting', 'running'} and
                (active['phase'] != 'running' or active.get('owner') == owner))


def release(store, owner):
    with portalocker.Lock(str(store.directory / 'reservation.lock'), timeout=0):
        with store.connect() as db:
            db.execute('DELETE FROM metadata WHERE key=?', (KEY + ':' + owner,))
        if read(store).get('owner') == owner:
            write(store, {'phase': 'idle'})


def running(store, owner):
    with portalocker.Lock(str(store.directory / 'reservation.lock'), timeout=0):
        active = read(store)
        state = _expire(store, request(store, owner), time.time())
        if (state.get('phase') not in {'waiting', 'running'} or
                active['phase'] == 'running' and active.get('owner') != owner):
            raise RuntimeError('Maintenance execution permit unavailable')
        state = dict(state, phase='running')
        save_request(store, state)
        write(store, state)


def blocker(store, task_class):
    from . import priority
    git = priority.read(store)
    if git['phase'] in {'checkout', 'blocked'} and task_class not in {'git', 'installation'}:
        return 'Git checkout requires verification'
    with portalocker.Lock(str(store.directory / 'reservation.lock'), timeout=0):
        state = read(store)
    if state['phase'] != 'running':
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
def leases(store, task_class, exclusive_groups=None):
    """Called under admission.lock; leases live for the entire execution scope."""
    group = group_for(task_class)
    groups = [group] if group in GROUPS else []
    exclusive = exclusive_groups is not None
    if exclusive:
        groups = sorted(set(exclusive_groups))
    if task_class.startswith('autofix:'):
        state = request(store, task_class)
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


def preferred_maintenance(store, task_class):
    """Called under admission.lock before taking leases; probes never retain locks."""
    if task_class != 'git' and not task_class.startswith('autofix:'):
        return None
    from . import install_request
    from .config import runtime_dir
    def available(groups, *, code_exclusive=False):
        try:
            with ExitStack() as stack:
                paths = [(store.directory / 'gate.lock', groups is None),
                         (runtime_dir() / 'git_update.lock', code_exclusive)]
                if groups is not None:
                    paths += [(store.directory / f'channel-{name}.lock', True) for name in sorted(groups)]
                if not code_exclusive and groups is None:
                    paths.append((store.directory / 'worker.lock', True))
                for path, exclusive in paths:
                    stream = stack.enter_context(path.open('a+'))
                    portalocker.lock(stream, (portalocker.LOCK_EX if exclusive else portalocker.LOCK_SH) | portalocker.LOCK_NB)
                return True
        except portalocker.LockException:
            return False
    installation = install_request.current()
    if installation.get('phase') == 'waiting' and available(installation.get('groups')):
        from .install_flow import check_business_tasks
        from .maintenance_gate import MaintenanceBusy
        try:
            check_business_tasks(installation.get('admission_plan', {}))
        except MaintenanceBusy:
            pass
        else:
            return 'An authorized installation is eligible to run first'
    git = request(store, 'git')
    if task_class != 'git' and git.get('phase') == 'waiting' and available(None, code_exclusive=True):
        from .git_update import training_blocker
        if not training_blocker():
            return 'Git maintenance is eligible to run first'
    return None
