"""Host-local fit admission and the opt-in unified training policy."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from contextvars import ContextVar
import json
import sys
import time
import uuid

import portalocker
import psutil

from src.automation.console import logger
from src.shared.file_ops import atomic_json
from .config import installed_config, runtime_dir

MODES = ('legacy', 'observe', 'unified')
PRIORITY = {'production': 0, 'manual': 1, 'promotion': 2, 'idle': 3}
_current: ContextVar[dict | None] = ContextVar('training_admission', default=None)


def available():
    return sys.platform == 'linux' and bool((installed_config() or {}).get('training_maintenance'))


def _fits_occupied():
    from src.proj.util.script.fit_lock import FitLock
    return FitLock.is_held() or FitLock.production_requested()


def _managed_alive(identity):
    from .runtime import RunStore, members
    run = RunStore().get(identity)
    if run.get('phase') not in ('starting', 'running', 'terminating', 'finalizing') or not run.get('pid'):
        return False
    try:
        return bool(members(run))
    except (OSError, psutil.Error, RuntimeError):
        return True


def _alive(row):
    try:
        process = psutil.Process(row['pid'])
        if process.create_time() == row['process_created'] and process.status() != psutil.STATUS_ZOMBIE:
            return True
    except psutil.NoSuchProcess:
        pass
    except (OSError, psutil.Error):
        return True
    return bool(row.get('managed_run') and row.get('phase') == 'active' and _managed_alive(row['managed_run']))


def _read(path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {'mode': 'legacy', 'target': 'legacy', 'requests': {}, 'observations': []}


def _reconcile(state):
    state['requests'] = {key: row for key, row in state['requests'].items() if _alive(row)}
    demand = state.get('promotion')
    if demand and demand['expires_at'] <= time.time():
        state.pop('promotion')
    if state['target'] == state['mode']:
        state['transition_blockers'] = []
        return
    blockers = [row['id'] for row in state['requests'].values() if row['phase'] == 'active']
    if not blockers and _fits_occupied():
        blockers = ['fit lock']
    state['transition_blockers'] = blockers
    if not blockers:
        state['mode'] = state['target']
        state.pop('promotion', None)


@contextmanager
def _transaction():
    directory = runtime_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'training-scheduler.json'
    with portalocker.Lock(str(directory / 'training-scheduler.lock'), timeout=10):
        state = _read(path)
        before = json.dumps(state)
        _reconcile(state)
        yield state
        if json.dumps(state) != before:
            atomic_json(path, state)


def policy():
    if not available():
        return 'legacy', False
    with _transaction() as state:
        return state['mode'], state['target'] != state['mode']


def _order(row):
    return PRIORITY[row['kind']], row['requested_at'], row['id']


def _conflict(left, right):
    if left['kind'] == right['kind'] == 'manual':
        return ('all' in (left['resource'], right['resource'])
                or left['resource'] == right['resource'] == 'nn')
    return True


def _blocker(state, row):
    others = list(state['requests'].values())
    if state.get('promotion') and row['kind'] == 'idle':
        others.append(state['promotion'])
    conflicts = [other for other in others if other['id'] != row['id'] and _conflict(row, other)
                 and (other['phase'] == 'active' or _order(other) < _order(row))]
    if conflicts:
        return min(conflicts, key=_order)
    return None


def _observe(state, row, point, blocker):
    detail = {'request': row['id'], 'kind': row['kind'], 'point': point,
              'decision': blocker['id'] if blocker else 'allow', 'label': row['label']}
    events = state['observations']
    # Deduplicate at each boundary so long fits do not evict earlier decisions.
    previous = next((event for event in reversed(events)
                     if event['request'] == row['id'] and event['point'] == point), None)
    if previous and detail == {key: previous[key] for key in detail}:
        return
    events.append(dict(detail, time=time.time()))
    del events[:-200]


@contextmanager
def bind(row):
    token = _current.set(row)
    try:
        yield
    finally:
        _current.reset(token)


def current():
    return _current.get()


def active_idle_runs():
    if not available():
        return ()
    with _transaction() as state:
        active = [row for row in state['requests'].values() if row['phase'] == 'active']
        if active and all(row['kind'] == 'idle' and row['managed_run'] for row in active):
            return tuple(row['managed_run'] for row in active)
    return ()


def automatic_kind(run):
    if run.get('task_kind') == 'model_promotion':
        return 'promotion'
    return run.get('training_priority', 'idle')


@contextmanager
def admit(kind, label, *, resource='all', request_id=None, requested_at=None, managed_run=None,
          checkpointable=False, wait=True, on_wait=None):
    """Register before acquiring any model or fit lock. Yield the admitted request."""
    if current() is not None or not available():
        yield current() or {}
        return
    process = psutil.Process()
    identity = request_id or uuid.uuid4().hex
    row = {'id': identity, 'kind': kind, 'label': label, 'resource': resource,
           'requested_at': requested_at if requested_at is not None else time.time(),
           'pid': process.pid, 'process_created': process.create_time(), 'managed_run': managed_run,
           'phase': 'waiting', 'checkpointable': checkpointable}
    started = time.monotonic()
    last_report = None
    try:
        while True:
            with _transaction() as state:
                existing = state['requests'].get(identity)
                if existing is not None and existing is not row and (
                        existing['pid'] != row['pid'] or existing['process_created'] != row['process_created']):
                    raise RuntimeError(f'Training request already owned: {identity}')
                state['requests'][identity] = row
                mode = state['mode']
                blocker = _blocker(state, row) if mode in ('unified', 'observe') else None
                if mode == 'observe':
                    _observe(state, row, 'admission', blocker)
                reason = None
                if state['target'] != mode:
                    reason = f'waiting for transition to {state["target"]}'
                elif mode == 'unified' and blocker:
                    reason = f'{blocker["kind"]} {blocker["label"]} ({blocker["id"]})'
                if reason:
                    row['blocked_by'] = reason
                else:
                    row.pop('blocked_by', None)
                    row.update(phase='active', mode=mode, wait_seconds=time.monotonic() - started if last_report else 0.)
                    if kind == 'promotion':
                        state.pop('promotion', None)
            if reason is None:
                with bind(row):
                    yield row
                return
            if not wait:
                yield None
                return
            now = time.monotonic()
            report_interval = 300 if kind == 'production' else 15
            if last_report is None or now - last_report >= report_interval:
                logger.info('Training request %s waiting %.0fs for %s', identity, now - started, reason)
                if on_wait is not None:
                    on_wait(now - started)
                last_report = now
            time.sleep(.5)
    finally:
        with _transaction() as state:
            existing = state['requests'].get(identity)
            if existing and existing['pid'] == row['pid'] and existing['process_created'] == row['process_created']:
                state['requests'].pop(identity)


@contextmanager
def managed_context(run_id):
    if available():
        mode_file = runtime_dir() / 'training-scheduler.json'
        row = _read(mode_file)['requests'].get(run_id)
        if row and row['phase'] == 'active' and row['managed_run'] == run_id:
            with bind(row):
                yield
            return
    yield


def pause_request():
    row = current()
    if not row or row['mode'] == 'legacy':
        return None
    with _transaction() as state:
        waiting = [other for other in state['requests'].values()
                   if other['phase'] == 'waiting' or row['mode'] == 'observe']
        if state.get('promotion'):
            waiting.append(state['promotion'])
        higher = [other for other in waiting if PRIORITY[other['kind']] < PRIORITY[row['kind']]
                  and _conflict(row, other)]
        blocker = min(higher, key=_order, default=None)
        if row['mode'] == 'observe':
            _observe(state, row, 'checkpoint', blocker)
        elif blocker and state['mode'] == state['target']:
            return {'kind': 'training_priority', 'id': blocker['id'], 'priority': blocker['kind']}
    return None


def discover_promotion(candidate, ready):
    """Publish a verified upgrade while an idle fit is still occupying the lane."""
    if not available():
        return
    with _transaction() as state:
        if state['mode'] not in ('unified', 'observe') or state['mode'] != state['target']:
            return
        if not candidate or not ready:
            state.pop('promotion', None)
            return
        identity = candidate['version']
        last = state.get('promotion') or {}
        timestamp = time.time()
        state['promotion'] = {
            'id': identity, 'label': candidate['task'], 'phase': 'waiting', 'kind': 'promotion', 'resource': 'all',
            'requested_at': last['requested_at'] if last.get('id') == identity else timestamp,
            'expires_at': timestamp + 300,
        }


def status(mode=None):
    with _transaction() as state:
        if mode is not None:
            state['target'] = mode
            _reconcile(state)
        result = dict(state)
    from .runtime import RunStore
    from .training_maintenance import candidates
    if available():
        store = RunStore()
        result['continuations'] = list(candidates(store))
        result['queued_automatic'] = [
            {key: row[key] for key in ('id', 'task', 'queued_at', 'blocked_by') if key in row}
            for row in store.owned('idle_worklist') if row['phase'] == 'queued'
        ]
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(prog='training-scheduler', description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    commands.add_parser('status')
    change = commands.add_parser('mode')
    change.add_argument('mode', choices=MODES)
    args = parser.parse_args(argv)
    if not available():
        parser.error('Requires installed Watchdog training maintenance on Linux')
    logger.info('%s', json.dumps(status(getattr(args, 'mode', None)), ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
