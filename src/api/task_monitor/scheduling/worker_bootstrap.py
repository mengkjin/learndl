"""Installed outside the checkout. Standard library only until code is locked.

Run as a file, never with -m: package imports must occur after flock. Both the
bootstrap and its settings are copied by the explicit scheduling installer.
"""
import fcntl
import json
import os
from pathlib import Path
import runpy
import sqlite3
import sys
import time


def code_pending(directory):
    with sqlite3.connect((directory / 'ledger.sqlite').as_uri() + '?mode=ro', uri=True) as db:
        row = db.execute("SELECT value FROM metadata WHERE key='scheduling_git'").fetchone()
    if row is None:
        raise RuntimeError('Missing scheduling priority state')
    state = json.loads(row[0])
    if state.get('phase') not in {'idle', 'draining', 'checkout', 'blocked'}:
        raise RuntimeError('Invalid scheduling priority state')
    return state['phase'] != 'idle'


def wait_for_code(directory, kind):
    try:
        return code_pending(directory)
    except (OSError, ValueError, sqlite3.Error, RuntimeError):
        # The independent control-failure outbox must survive ledger loss.
        # This exception applies only to mail, still under the physical code
        # read lock; every data/repair launcher continues to fail closed.
        if kind == 'email_outbox':
            return False
        raise


def main():
    directory = Path(__file__).resolve().parent
    settings = json.loads((directory / 'worker-settings.json').read_text())
    kind, *args = sys.argv[1:]
    if kind == 'calendar':
        os.environ['LEARNDL_SCHEDULE_SLOT'] = str(int(time.time() // 60))
    targets = {
        'autofix': ('src.api.task_monitor.scheduling.autofix', ['worker']),
        'deferred': ('src.api.task_monitor.scheduling.runner', ['deferred']),
        'calendar': ('src.api.task_monitor.scheduling.runner', ['run', *args]),
        'training': ('src.api.task_monitor.scheduling.idle_worklist', ['worker']),
        'email_outbox': ('src.api.task_monitor.scheduling.auxiliary', ['email_outbox']),
        'cli_recovery': ('src.api.task_monitor.scheduling.auxiliary', ['cli_recovery']),
        'task_monitor_cache': ('src.api.task_monitor.scheduling.auxiliary', ['task_monitor_cache']),
    }
    module, arguments = targets[kind]
    runtime = Path(settings['runtime'])
    config = json.loads((runtime / 'installed.json').read_text())
    modern = config.get('scheduler_version', 1) >= 2
    if kind == 'calendar' and modern:
        if len(args) != 1 or args[0] not in config['tasks']:
            raise ValueError('Unknown installed calendar task')
        # Persist before even waiting for a code lease: a timer firing during
        # checkout must survive launcher death or a partially updated tree.
        with (directory / 'admission.lock').open('r+') as admission:
            fcntl.flock(admission, fcntl.LOCK_EX)
            with sqlite3.connect((directory / 'ledger.sqlite').as_uri() + '?mode=rw', uri=True) as db:
                db.execute('INSERT OR IGNORE INTO scheduled VALUES (?,?,?,?)',
                           (args[0] + ':' + os.environ['LEARNDL_SCHEDULE_SLOT'], args[0], 'queued', time.time()))
    auxiliary = kind in {'email_outbox', 'cli_recovery', 'task_monitor_cache'}
    if auxiliary and not modern:
        return 0
    # Check before taking SH as well: otherwise synchronized helper timers can
    # repeatedly steal a short read lease from an already waiting Git writer.
    if modern and wait_for_code(directory, kind):
        return 0
    with (runtime / 'git_update.lock').open('a') as guard:
        try:
            fcntl.flock(guard, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0
        if modern and wait_for_code(directory, kind):
            return 0
        os.chdir(settings['root'])
        sys.path.insert(0, settings['root'])
        sys.argv = [module, *arguments]
        runpy.run_module(module, run_name='__main__')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
