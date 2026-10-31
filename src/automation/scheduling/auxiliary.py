"""Independent bounded helper services and the sole scheduler mail consumer."""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import time

import portalocker

from .config import installed_config, runtime_dir

NAMES = ('email_outbox', 'cli_recovery', 'task_monitor_cache')


def queue():
    path = runtime_dir() / 'watchdog-notices.sqlite'
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=10)
    db.executescript('''
        CREATE TABLE IF NOT EXISTS snapshots (id INTEGER PRIMARY KEY AUTOINCREMENT, data TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS errors (id TEXT PRIMARY KEY, detail TEXT NOT NULL, sent INTEGER NOT NULL DEFAULT 0);
    ''')
    return db


def publish(state, failures, probed):
    with queue() as db:
        db.execute('INSERT INTO snapshots(data) VALUES (?)',
                   (json.dumps({'alerts': state['alerts'], 'failures': failures, 'probed': probed}),))


def control_error(exc):
    detail = f'{type(exc).__name__}: {exc}'
    with queue() as db:
        db.execute('INSERT OR IGNORE INTO errors(id,detail) VALUES (?,?)',
                   (hashlib.sha256(detail.encode()).hexdigest(), detail))


def deliver(context, sender):
    from src.automation.watchdog import _load_state, _save_state, _deliver_alerts
    path = runtime_dir() / 'auxiliary' / 'alerts.json'
    state = _load_state(path)
    with queue() as db:
        rows = db.execute('SELECT id,data FROM snapshots WHERE id>? ORDER BY id',
                          (state.get('snapshot', 0),)).fetchall()
    for sequence, raw in rows:
        packet = json.loads(raw)
        alerts = state['alerts']
        if not state.get('snapshot'):
            alerts['unit_alerted'] = dict(packet['alerts'].get('unit_alerted', {}))
        alerts['pending_tasks'] = sorted(set(alerts['pending_tasks']) | set(packet['alerts']['pending_tasks']))
        # Seed historical receipts when migrating from the synchronous consumer.
        alerts['task_alerted'] = sorted(set(alerts.get('task_alerted', [])) | set(packet['alerts'].get('task_alerted', [])))
        if packet['probed']:
            failed = {unit for unit, _ in packet['failures']}
            for unit in alerts.get('unit_alerted', {}):
                if unit not in failed:
                    alerts['unit_alerted'][unit] = False
            state['failures'] = packet['failures']
        state['snapshot'] = sequence
    _save_state(path, state)
    # Receipts and cursor are owned only by this single-instance mail worker.
    failures = []
    try:
        if not _deliver_alerts(task_db=context.task_db, state=state, unit_failures=state.get('failures', []),
                               state_path=path, email_sender=sender):
            failures.append('watchdog notification delivery pending')
    except Exception as exc:
        failures.append(f'{type(exc).__name__}: {exc}')
    with queue() as db:
        db.execute('DELETE FROM snapshots WHERE id<=?', (state.get('snapshot', 0),))
        errors = db.execute('SELECT id,detail FROM errors WHERE sent=0').fetchall()
    for key, detail in errors:
        try:
            if sender('Autofix control failure', 'Autofix is paused; restore the existing control state.\n' + detail,
                      queue_on_failure=False, delivery_id='autofix-control:' + key):
                with queue() as db:
                    db.execute('UPDATE errors SET sent=1 WHERE id=?', (key,))
        except Exception as exc:
            failures.append(f'{type(exc).__name__}: {exc}')
    from .runtime import deliver_timeout_events
    from .install_request import deliver as deliver_installation
    from .git_update import deliver_update_emails
    from .autofix import deliver as deliver_autofix
    from .maintenance_gate import configured
    from src.api.task_monitor.interactive import maintain
    from src.automation.cli_recovery import deliver as deliver_cli
    # An unavailable queue must not prevent delivery from the remaining queues.
    callbacks = [lambda: deliver_timeout_events(sender), lambda: deliver_installation(sender),
                 lambda: deliver_update_emails(sender), lambda: maintain(context.task_db, sender),
                 lambda: deliver_cli(sender), lambda: deliver_autofix(sender) if configured() else True]
    for callback in callbacks:
        try:
            if callback() is False:
                failures.append('notification delivery pending')
        except Exception as exc:
            failures.append(f'{type(exc).__name__}: {exc}')
    if failures:
        raise RuntimeError('; '.join(failures))


def worker(name):
    from .priority import enabled
    if not enabled():
        return 0
    from src.automation.watchdog import (_WatchdogContext, _load_state, _save_state,
                                              _run_cli_recovery, _run_task_monitor_cache, _run_email_outbox)
    from src.api.task_monitor.output import OutputCache
    from src.api.util.backend.task import TaskDatabase
    from src.proj.util.web.emailer import Email
    directory = runtime_dir() / 'auxiliary'
    directory.mkdir(parents=True, exist_ok=True)
    try:
        with portalocker.Lock(str(directory / f'{name}.lock'), timeout=0):
            path = directory / f'{name}.json'
            state = _load_state(path)
            config = installed_config()['watchdog']
            options = config['jobs'].get(name, {})
            now = time.time()
            due = options.get('enabled', False) and now >= state.get('last_job_attempt', 0) + options.get('interval_seconds', 60)
            if not due and name != 'email_outbox':
                return 0
            state.update(status='running', last_attempt=now)
            _save_state(path, state)
            context = _WatchdogContext(TaskDatabase(), OutputCache(), [], now, lambda _: (True, ''), config)
            errors = []
            try:
                if due:
                    state['last_job_attempt'] = now
                    _save_state(path, state)
                    result = {'email_outbox': _run_email_outbox, 'cli_recovery': _run_cli_recovery,
                              'task_monitor_cache': _run_task_monitor_cache}[name](context)
                    state['stats'] = result.stats
            except Exception as exc:
                errors.append(f'{type(exc).__name__}: {exc}')
            if name == 'email_outbox':
                try:
                    deliver(context, Email.send)
                except Exception as exc:
                    errors.append(f'{type(exc).__name__}: {exc}')
            state.update(status='error' if errors else 'complete', error='; '.join(errors) or None,
                         finished_at=time.time())
            state['last_error'] = state['error']
            if not errors:
                state['last_success_at'] = state['finished_at']
            _save_state(path, state)
            return int(bool(errors))
    except portalocker.LockException:
        return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job', choices=NAMES)
    raise SystemExit(worker(parser.parse_args().job))
