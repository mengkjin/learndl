"""Read-only, bounded evidence snapshots for delayed maintenance notifications."""
from datetime import datetime, timezone
import json
import socket
import sqlite3
import time

import portalocker
import psutil

from .config import runtime_dir
from .autofix_store import state_dir

PREFIX = 'maintenance-snapshot:'


def stamp(value):
    return datetime.fromtimestamp(value, timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC') if value is not None else 'unknown'


def duration(value):
    minutes = max(0, int(value // 60))
    return f'{minutes // 60}h {minutes % 60}m'


def capture(state):
    now = time.time()
    result = dict(state=state, captured_at=now, host=socket.gethostname(), tasks=[], limitations=[])
    deadline = time.monotonic() + 2
    sources = [(runtime_dir() / 'runs.sqlite', 'runs', ('starting', 'running', 'terminating', 'finalizing')),
               (state_dir() / 'ledger.sqlite', 'fixes', ('starting', 'running'))]
    for path, table, phases in sources:
        if not path.exists():
            result['limitations'].append(f'Missing ledger: {path.name}')
            continue
        try:
            with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, timeout=.1) as db:
                db.set_progress_handler(lambda: int(time.monotonic() > deadline), 1000)
                rows = db.execute(f'SELECT data FROM {table} WHERE json_extract(data,\'$.phase\') IN ({",".join("?" for _ in phases)}) LIMIT 101', phases)
                for (raw,) in rows:
                    if time.monotonic() > deadline:
                        result['limitations'].append('Diagnostic time budget reached')
                        break
                    run = json.loads(raw)
                    group = 'autofix' if table == 'fixes' else 'research' if run.get('owner') == 'idle_worklist' else run.get('channel', 'data')
                    groups = state.get('groups')
                    if groups is not None and group not in groups and table != 'fixes':
                        continue
                    status = 'unverified'
                    try:
                        proc = psutil.Process(run['pid'])
                        status = 'alive' if abs(proc.create_time()-run['process_created']) < .01 and proc.status() != psutil.STATUS_ZOMBIE else 'stale identity'
                    except psutil.NoSuchProcess:
                        status = 'root exited; descendants or unreconciled state may remain'
                    except (psutil.Error, KeyError, TypeError):
                        pass
                    result['tasks'].append(dict(id=run['id'], task=run.get('task', run['id']), pid=run.get('pid'),
                        channel=group, started_at=run.get('started_at'), status=status, log=run.get('log_path')))
                    if len(result['tasks']) >= 100:
                        result['limitations'].append('Task list capped at 100 records')
                        break
        except (sqlite3.Error, OSError, ValueError) as exc:
            result['limitations'].append(f'{path.name}: {type(exc).__name__}')
    return result


def render(snapshot):
    state, now = snapshot['state'], snapshot['captured_at']
    owner = state.get('owner', 'installation')
    operation = 'Automatic Git update (origin/master)' if owner == 'git' else 'Watchdog installation' if owner == 'installation' else f'Autofix request {owner.removeprefix("autofix:")}'
    started = state.get('requested_at', state.get('created_at'))
    lines = [f'Host: {snapshot["host"]}', f'Operation: {operation}',
             f'Request: {state.get("id", owner)}', f'Evidence captured: {stamp(now)}',
             f'Waiting since: {stamp(started)}', f'Total wait: {duration(now-started) if started is not None else "unknown"}',
             f'Status: {state.get("phase", "waiting")}', f'Affected channels: {state.get("groups") or "all / not recorded"}',
             f'Last admission failure: {state.get("last_failure", "not recorded by this worker")}',
             f'Next eligible attempt: {stamp(state.get("cooldown_until"))}',
             '', 'Related task candidates (not proof of lock ownership):']
    for run in snapshot['tasks'][:10]:
        lines += [f'- {run["task"]} | run {run["id"]} | PID {run["pid"]} | {run["channel"]}',
                  f'  Started: {stamp(run["started_at"])}; elapsed: {duration(now-run["started_at"]) if run["started_at"] is not None else "unknown"}',
                  f'  Identity: {run["status"]}; log: {run["log"] or "not recorded"}']
    if not snapshot['tasks']:
        lines.append('No matching managed task identified; an unmanaged process, helper, or control-state condition may be responsible.')
    lines += snapshot['limitations']
    lines += ['', 'This is a captured observation, not the state at email delivery time.',
              'Waiting does not terminate existing work or reserve business channels. This timeout is not evidence of ledger corruption.']
    return 'Maintenance admission delayed - ' + operation, '\n'.join(lines)


def record_failure(store, owner, error):
    try:
        _record_failure(store, owner, error)
    except (OSError, RuntimeError, sqlite3.Error, portalocker.LockException):
        pass  # Diagnostics never change admission outcomes.


def _record_failure(store, owner, error):
    from . import channels
    with portalocker.Lock(str(store.directory / 'reservation.lock'), timeout=0):
        state = channels.request(store, owner)
        if state:
            state['last_failure'] = str(error)
            channels.save_request(store, state)


def attachment(snapshot):
    if len(snapshot['tasks']) <= 10:
        return []
    import hashlib
    from .installer import atomic_json
    payload = json.dumps(snapshot, sort_keys=True)
    directory = runtime_dir() / 'maintenance-notices'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (hashlib.sha256(payload.encode()).hexdigest() + '.json')
    atomic_json(path, snapshot)
    return [path]
