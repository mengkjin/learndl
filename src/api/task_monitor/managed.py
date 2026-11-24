"""Read-only monitor adapter for scheduler and autofix process groups."""
from __future__ import annotations

__all__ = [
    'PHASES',
    'rows',
    'snapshots',
]
import json
import re
import sqlite3
from pathlib import Path
import psutil

PHASES = {'queued':'starting', 'starting':'starting', 'running':'running',
          'terminating':'running', 'finalizing':'running', 'complete':'complete',
          'success':'complete', 'error':'error', 'failed':'error', 'timed_out':'error',
          'unknown':'error', 'killed':'killed', 'cancelled':'killed'}
SCHEDULE_LOG_PREFIX_BYTES = 8192
_SCHEDULE_SCRIPT = re.compile(r'4_train/2_schedule_model\.py@\d+')
_SCHEDULE_CAPTURE = re.compile(
    rb'CrashProtectorCatcher (?:>>|&gt;&gt;) task_id="(4_train/2_schedule_model\.py@\d+)", Capturing Start'
)


def duplicate_schedule_wrappers(tasks, read_prefix, has_script_output):
    """Find ordinary worklist wrappers whose log explicitly names the recorded training script."""
    scripts = {task['task_id']: task for task in tasks if _SCHEDULE_SCRIPT.fullmatch(task['task_id'])}
    if not scripts:
        return set()
    duplicates = set()
    for task in tasks:
        if (not task['task_id'].startswith('managed:scheduler:')
                or task.get('cmd') != 'watchdog / idle_worklist / scheduler'):
            continue
        matches = set(_SCHEDULE_CAPTURE.findall(read_prefix(task)))
        if len(matches) != 1:
            continue
        script = scripts.get(next(iter(matches)).decode('ascii'))
        if (script and script['status'] == task['status']
                and script.get('pid_alive') is not False and has_script_output(script)):
            duplicates.add(task['task_id'])
    return duplicates


def rows(path, query, parameters=(), *, strict=False):
    if not path.is_file():
        return []
    try:
        with sqlite3.connect(path.resolve().as_uri()+'?mode=ro', uri=True, timeout=2) as db:
            db.execute('PRAGMA query_only=ON')
            return db.execute(query, parameters).fetchall()
    except (sqlite3.Error, OSError):
        if strict:
            raise
        return []


def snapshots(repository, *, detail=False, task_id=None, since=None, hidden=None):
    from .core import TaskSnapshot
    scheduler = repository.runtime_dir/'scheduling/runs.sqlite'
    links = {}
    for linked_task_id, run_id in rows(scheduler, 'SELECT task_id,run_id FROM task_links', strict=since is not None):
        links.setdefault(run_id, set()).add(linked_task_id)
    # Metadata only. Never instantiate either lifecycle store (constructors may write).
    scripts = {r[0]: r[1:] for r in rows(repository.db_path,
        "SELECT task_id,pid,status,start_time FROM task_records" +
        (" WHERE status IN ('starting','running') OR end_time>=?" if since is not None else ''),
        (since,) if since is not None else (), strict=since is not None)}
    sources = [('scheduler', scheduler, 'SELECT data FROM runs')]
    if repository.autofix_path is not None:
        sources.append(('autofix', repository.autofix_path, 'SELECT data FROM fixes'))
    result = []
    for kind, path, query in sources:
        parameters = ()
        if task_id is not None:
            prefix = f'managed:{kind}:'
            if not task_id.startswith(prefix):
                continue
            if kind == 'scheduler':
                query += ' WHERE id=?'
                parameters = (task_id.removeprefix(prefix),)
        elif since is not None:
            query += " WHERE json_extract(data, '$.phase') IN ('queued','starting','running','terminating','finalizing') OR COALESCE(json_extract(data, '$.finished_at'),0)>=?"
            parameters = (since,)
        for (raw,) in rows(path, query, parameters, strict=since is not None):
            try:
                doc = json.loads(raw)
                if not isinstance(doc, dict):
                    continue
                if not doc.get('id'):
                    continue
                identity = f'managed:{kind}:{doc["id"]}'
                if task_id is not None and task_id != identity:
                    continue
                phase = doc.get('phase')
                if phase not in PHASES:
                    if hidden is not None and phase == 'deferred':
                        hidden.add(identity)
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
                    if hidden is not None:
                        hidden.add(identity)
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
                        if detail and alive and since is None:
                            children = tuple(p.pid for p in proc.children(recursive=True))
                    except psutil.NoSuchProcess:
                        alive = False
                    except psutil.Error:
                        alive = None
                result.append(TaskSnapshot(
                    task_id=identity,
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
