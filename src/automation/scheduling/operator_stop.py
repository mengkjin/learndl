"""Explicit, identity-bound operator stop of a supervised process group."""
import json
import signal
import time
from .config import runtime_dir
from .runtime import RunStore, members, signal_members
from .force_install import IDENTITY


def store_for(kind):
    if kind == 'scheduler':
        path=runtime_dir()/'runs.sqlite'
        if not path.is_file():
            raise ValueError('Scheduler ledger is missing')
        return RunStore(path)
    raise ValueError('This managed task type does not support Monitor stop yet')


def prepare(task_id, pid, created):
    prefix,kind,run_id=task_id.split(':',2)
    if prefix!='managed':
        raise ValueError('Invalid managed identity')
    run=store_for(kind).get(run_id)
    if not run or run['phase'] not in {'running','starting','terminating'}:
        raise ValueError('Managed task has already finished')
    if run.get('pid')!=pid or run.get('process_created')!=created or not members(run):
        raise ValueError('Managed process identity changed')
    return {key:run.get(key) for key in IDENTITY}


def stop(task_id, identity):
    _,kind,run_id=task_id.split(':',2)
    store=store_for(kind)
    run=store.get(run_id)
    if not run or {key:run.get(key) for key in IDENTITY}!=identity:
        raise ValueError('Managed process identity changed since confirmation')
    processes=members(run)
    if not processes:
        raise ValueError('Managed group has already exited')
    with store.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        current=json.loads(db.execute('SELECT data FROM runs WHERE id=?',(run_id,)).fetchone()[0])
        if current['phase'] not in {'running','starting','terminating'} or {key:current.get(key) for key in IDENTITY}!=identity:
            raise ValueError('Managed task changed since confirmation')
        intent=dict(operator_stop='Learndl Monitor',suppress_auto_recovery=True,term_at=time.time())
        # The existing durable stop overlay survives stale writes by the live runner.
        db.execute('INSERT OR IGNORE INTO installation_stops VALUES (?,?)',(run_id,json.dumps(intent)))
        current.update(intent,phase='terminating')
        db.execute('UPDATE runs SET data=? WHERE id=?',(json.dumps(current),run_id))
        for child_id,raw in db.execute("SELECT id,data FROM runs WHERE json_extract(data,'$.phase')='queued'").fetchall():
            child=json.loads(raw)
            if child.get('parent_run_id')==run_id:
                child_intent={**intent,'phase':'cancelled'}
                db.execute('INSERT OR IGNORE INTO installation_stops VALUES (?,?)',(child_id,json.dumps(child_intent)))
                child.update(child_intent)
                db.execute('UPDATE runs SET data=? WHERE id=?',(json.dumps(child),child_id))
    # Watchdog retains ownership and escalates to KILL after the normal grace period.
    signal_members(current,processes,signal.SIGTERM)
    return 'Stop requested for the entire managed process group; automatic recovery disabled.'
