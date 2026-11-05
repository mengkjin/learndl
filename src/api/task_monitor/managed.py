"""Read-only monitor adapter for scheduler and autofix process groups."""
from __future__ import annotations

__all__ = [
    'PHASES',
    'rows',
    'snapshots',
]
import json
import sqlite3
from pathlib import Path
import psutil

PHASES = {'queued':'starting', 'starting':'starting', 'running':'running',
          'terminating':'running', 'finalizing':'running', 'complete':'complete',
          'success':'complete', 'error':'error', 'failed':'error', 'timed_out':'error',
          'unknown':'error', 'killed':'killed', 'cancelled':'killed'}


def rows(path, query):
    if not path.is_file():
        return []
    try:
        with sqlite3.connect(path.resolve().as_uri()+'?mode=ro', uri=True, timeout=2) as db:
            db.execute('PRAGMA query_only=ON')
            return db.execute(query).fetchall()
    except (sqlite3.Error, OSError):
        return []


def snapshots(repository, *, detail=False, task_id=None):
    from .core import TaskSnapshot
    scheduler = repository.runtime_dir/'scheduling/runs.sqlite'
    links = {}
    for linked_task_id, run_id in rows(scheduler, 'SELECT task_id,run_id FROM task_links'):
        links.setdefault(run_id, set()).add(linked_task_id)
    # Metadata only. Never instantiate either lifecycle store (constructors may write).
    scripts = {r[0]: r[1:] for r in rows(repository.db_path,
        'SELECT task_id,pid,status,start_time FROM task_records')}
    sources = [('scheduler', scheduler, 'SELECT data FROM runs')]
    if repository.autofix_path is not None:
        sources.append(('autofix', repository.autofix_path, 'SELECT data FROM fixes'))
    result = []
    for kind, path, query in sources:
        for (raw,) in rows(path, query):
            try:
                doc = json.loads(raw)
                if not isinstance(doc, dict):
                    continue
                if task_id is not None and task_id != f'managed:{kind}:{doc.get("id")}':
                    continue
                phase = doc.get('phase')
                if phase not in PHASES or not doc.get('id'):
                    continue
                status = PHASES[phase]
                pid = doc.get('pid')
                created = doc.get('process_created')
                # Only collapse the SAME process explicitly linked to a script.
                # Child scripts have different PIDs and remain separate from their group.
                if kind == 'scheduler' and pid and any(
                    key in scripts and scripts[key][0] == pid
                    and ('running' if scripts[key][1] in ('starting','running') else scripts[key][1])
                        == ('running' if status in ('starting','running') else status)
                    for key in links.get(doc['id'], ())):
                    continue
                item = doc.get('item', {})
                started = doc.get('started_at')
                queued = doc.get('queued_at') or doc.get('created_at') or doc.get('registered_at') or started or doc.get('finished_at') or 0
                log = Path(doc['log_path']) if doc.get('log_path') else None
                alive = None
                children = ()
                if status in ('starting','running') and pid and created:
                    try:
                        proc = psutil.Process(pid)
                        alive = abs(proc.create_time()-created) < .01 and proc.is_running() and proc.status()!=psutil.STATUS_ZOMBIE
                        if detail and alive:
                            children = tuple(p.pid for p in proc.children(recursive=True))
                    except psutil.NoSuchProcess:
                        alive = False
                    except psutil.Error:
                        alive = None
                result.append(TaskSnapshot(
                    task_id=f'managed:{kind}:{doc["id"]}',
                    script=doc.get('script') or item.get('script') or doc.get('task') or doc['id'],
                    cmd=f'watchdog / {doc.get("owner",kind)} / {doc.get("task_kind",kind)}',
                    create_time=float(queued), status=status, source='watchdog', pid=pid,
                    start_time=started, end_time=doc.get('finished_at'), exit_code=doc.get('returncode', doc.get('code')),
                    exit_message=f'Managed phase: {phase}; '+str(doc.get('blocked_by') or ''),
                    exit_error=doc.get('error'), exit_files=(log,) if detail and log else (),
                    pid_alive=alive, managed_kind=kind, managed_phase=phase,
                    process_created=created, child_pids=children))
            except (TypeError, ValueError, KeyError):
                # One malformed row must not hide the remaining managed processes.
                continue
    return result
