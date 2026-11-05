"""Stop only previewed, identity-verified managed scopes before exclusive installation."""
from __future__ import annotations

__all__ = [
    'IDENTITY',
    'targets',
    'mark_run',
    'drain',
]

import json
import signal
import time

import portalocker

from .autofix_store import Store, state_dir
from .runtime import RunStore, members, signal_members
from .maintenance_gate import MaintenanceBusy

IDENTITY = ('pid', 'process_created', 'runner_pid', 'uid', 'boot_id', 'cgroup')


def targets(plan):
    impact = plan.get('impact', {})
    if impact.get('mode') in {'config', 'compatible'}:
        return []
    groups = impact.get('groups')
    result = []
    for run in RunStore().active():
        group = 'research' if run.get('owner') == 'idle_worklist' else run.get('channel', 'data')
        if groups is not None and group not in groups:
            continue
        if not run.get('pid'):
            raise MaintenanceBusy('A managed launcher is still registering its process; preview again')
        processes = members(run)  # Invalid/reused identities fail before authorization.
        if processes:
            result.append(dict(kind='run', id=run['id'], task=run['task'], started_at=run['started_at'],
                               elapsed_seconds=max(0, time.time()-run['started_at']), identity={key: run.get(key) for key in IDENTITY}))
    for row in Store(state_dir()).rows():
        if row['phase'] not in {'starting', 'running'}:
            continue
        blocking = row['item'].get('blocking', 'none')
        affected = {'autofix'} | ({'research', 'data'} if blocking == 'research_and_data' else
                                {'research'} if blocking == 'research' else set())
        if groups is not None and not affected.intersection(groups):
            continue
        if not row.get('pid'):
            raise MaintenanceBusy('An autofix launcher is still registering; preview again')
        if members(row):
            result.append(dict(kind='autofix', id=row['id'], task=row['id'], started_at=row['started_at'],
                               elapsed_seconds=max(0, time.time()-row['started_at']), identity={key: row.get(key) for key in IDENTITY}))
    return result


def _same(a, b):
    return (a['kind'], a['id'], a['identity']) == (b['kind'], b['id'], b['identity'])


def mark_run(store, run_id, installation_id):
    with store.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        row = db.execute('SELECT data FROM runs WHERE id=?', (run_id,)).fetchone()
        run = json.loads(row[0])
        if run['phase'] not in {'starting', 'running', 'terminating', 'finalizing'}:
            return run
        run.update(installation_stop=installation_id, suppress_auto_recovery=True,
                   phase='terminating', term_at=run.get('term_at', time.time()))
        stop = {key: run[key] for key in ('installation_stop', 'suppress_auto_recovery', 'term_at')}
        db.execute('INSERT OR IGNORE INTO installation_stops VALUES (?,?)', (run_id, json.dumps(stop)))
        db.execute('UPDATE runs SET data=? WHERE id=?', (json.dumps(run), run_id))
        # Queued children derived from this attempt must not revive it after draining.
        for key, data in db.execute("SELECT id,data FROM runs WHERE json_extract(data,'$.phase')='queued'").fetchall():
            child = json.loads(data)
            if child.get('parent_run_id') == run_id:
                child.update(phase='cancelled', installation_stop=installation_id, suppress_auto_recovery=True)
                db.execute('INSERT OR IGNORE INTO installation_stops VALUES (?,?)',
                           (key, json.dumps(dict(installation_stop=installation_id, suppress_auto_recovery=True, phase='cancelled'))))
                db.execute('UPDATE runs SET data=? WHERE id=?', (json.dumps(child), key))
    return run


def drain(approval):
    from . import install_request as requests
    authorized = approval['force_targets']
    with portalocker.Lock(str(state_dir() / 'admission.lock'), mode='r+', timeout=0):
        actual = targets(approval['plan'])
        if any(not any(_same(item, allowed) for allowed in authorized) for item in actual):
            raise MaintenanceBusy('Managed task identities changed since preview; authorize again')
        with requests.lock():
            row = requests.current()
            if row.get('id') != approval['id'] or row.get('phase') != 'waiting':
                raise MaintenanceBusy('Force request is no longer waiting')
            row.update(phase='draining', groups=approval['plan'].get('impact', {}).get('groups'))
            requests.save(row)
    runs, fixes = RunStore(), Store(state_dir())
    started = time.monotonic()
    while True:
        survivors = False
        for target in actual:
            run = runs.get(target['id']) if target['kind'] == 'run' else fixes.get(target['id'])
            if run is None or {key: run.get(key) for key in IDENTITY} != target['identity']:
                raise RuntimeError('Managed process identity changed during drain')
            processes = members(run)
            if not processes:
                continue  # Natural completion keeps its own result.
            survivors = True
            if target['kind'] == 'run':
                run = mark_run(runs, run['id'], approval['id'])
            else:
                fixes.update(run['id'], installation_stop=approval['id'], suppress_auto_recovery=True)
                from . import autofix_mailbox
                if run.get('channel') and run.get('nonce'):
                    from pathlib import Path
                    autofix_mailbox.write(Path(run['channel']), 'terminate.json', {'nonce': run['nonce']})
            elapsed = time.monotonic() - started
            if elapsed >= 120:
                raise MaintenanceBusy('Force drain exceeded 120 seconds; surviving processes remain protected')
            if elapsed >= 30 and target['kind'] == 'run':
                run['kill_at'] = run.get('kill_at', time.time())
                runs.put(run)
            signal_members(run, processes, signal.SIGTERM if elapsed < 30 else signal.SIGKILL)
        if not survivors:
            return
        time.sleep(1)
