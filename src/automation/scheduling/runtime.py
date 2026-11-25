"""Durable run identities and two-phase termination of installer-owned runs."""
from __future__ import annotations

__all__ = [
    'RunStore',
    'record_script_email',
    'idle_exit_needs_email',
    'register_task',
    'timeout_override',
    'process_cgroup',
    'termination_requested',
    'in_run_scope',
    'boot_id',
    'previous_boot',
    'members',
    'signal_members',
    'timeout_message',
    'timeout_due',
    'finish_idle_event',
    'inspect_timeouts',
    'deliver_timeout_events',
    'timeout_task_ids',
    'operator_task_stop_ids',
    'pin_crash_logs',
]

import json
import os
import signal
import sqlite3
import time
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any

import psutil

from .config import runtime_dir

if TYPE_CHECKING:
    from src.api.util.backend.task import TaskItem

OPERATOR_STOP_EVENT = 'operator_stopped'
OPERATOR_TASK_PREFIX = 'task-center-stop:'
_AWAITING_OPERATOR_EXIT = -2


class RunStore:
    def __init__(self, path: Path | None = None):
        self.path = path or runtime_dir() / 'runs.sqlite'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript('''
                CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS installation_stops (id TEXT PRIMARY KEY, data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS task_links (task_id TEXT PRIMARY KEY, run_id TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS events (id TEXT PRIMARY KEY, body TEXT NOT NULL, sent INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS script_mail (
                    run_id TEXT NOT NULL, outcome TEXT NOT NULL, sent_at REAL NOT NULL,
                    PRIMARY KEY(run_id, outcome)
                );
                CREATE TABLE IF NOT EXISTS run_evidence (run_id TEXT PRIMARY KEY, data TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS runs_phase ON runs(json_extract(data, '$.phase'));
            ''')

    def connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=15)

    def put(self, run: dict) -> None:
        with self.connect() as connection:
            stop = connection.execute('SELECT data FROM installation_stops WHERE id=?', (run['id'],)).fetchone()
            if stop:
                run = dict(run, **json.loads(stop[0]))
            connection.execute('INSERT INTO runs VALUES (?, ?) ON CONFLICT(id) DO UPDATE SET data=excluded.data', (run['id'], json.dumps(run)))

    def get(self, run_id: str) -> dict:
        with self.connect() as connection:
            row = connection.execute('SELECT data FROM runs WHERE id=?', (run_id,)).fetchone()
        return self._with_stop(json.loads(row[0])) if row else {}

    def active(self) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute("SELECT data FROM runs WHERE json_extract(data, '$.phase') IN ('starting','running','terminating','finalizing')").fetchall()
        return [self._with_stop(json.loads(row[0])) for row in rows]

    def owned(self, owner: str) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute("SELECT data FROM runs WHERE json_extract(data, '$.owner')=? ORDER BY rowid", (owner,)).fetchall()
        return [self._with_stop(json.loads(row[0])) for row in rows]

    def _with_stop(self, run):
        with self.connect() as db:
            stop = db.execute('SELECT data FROM installation_stops WHERE id=?', (run['id'],)).fetchone()
        return dict(run, **json.loads(stop[0])) if stop else run

    def link(self, task_id: str, run_id: str) -> None:
        with self.connect() as connection:
            connection.execute('INSERT OR IGNORE INTO task_links VALUES (?, ?)', (task_id, run_id))

    def links(self, run_id: str) -> list[str]:
        with self.connect() as connection:
            return [row[0] for row in connection.execute('SELECT task_id FROM task_links WHERE run_id=?', (run_id,))]

    def event(self, run: dict, kind: str, detail: str, *, awaiting_exit: bool = False) -> bool:
        body = json.dumps({'run': run, 'detail': detail, 'kind': kind})
        with self.connect() as connection:
            inserted = connection.execute('INSERT OR IGNORE INTO events(id,body,sent) VALUES (?,?,?)',
                                          (f'{run["id"]}:{kind}', body, _AWAITING_OPERATOR_EXIT if awaiting_exit else 0))
            return inserted.rowcount == 1

    def finish_operator_task(self, run_id: str, finished_at: float | None,
                             logs: dict[str, list[str]] | None = None) -> None:
        event_id = f'{run_id}:{OPERATOR_STOP_EVENT}'
        with self.connect() as connection:
            if finished_at is not None:
                connection.execute("""
                    UPDATE events SET sent=0, body=json_set(body, '$.run.finished_at', ?, '$.run.operator_crash_logs',
                        coalesce(json(?), json_extract(body, '$.run.operator_crash_logs')))
                    WHERE id=? AND sent=?
                """, (finished_at, json.dumps(logs) if logs is not None else None, event_id, _AWAITING_OPERATOR_EXIT))
            else:
                connection.execute('DELETE FROM events WHERE id=? AND sent=?', (event_id, _AWAITING_OPERATOR_EXIT))


def pin_crash_logs(store: RunStore, task: TaskItem) -> list[str]:
    """Keep the same inode so writes through the original handle are captured until exit."""
    key = sha256(task.id.encode()).hexdigest()
    folder = store.path.parent / 'operator_stops' / key
    paths = []
    for raw in task.get_crash_protector():
        source = Path(raw)
        link = folder / source.name
        try:
            folder.mkdir(mode=0o700, parents=True, exist_ok=True)
            os.link(source, link)
        except FileExistsError:
            pass
        except OSError:
            # Captures may already be closing. The exit handler can find any
            # surviving original and retry the link after the process exits.
            continue
        paths.append(str(link))
    return paths


def record_script_email(*, success: bool) -> None:
    """Receipt after SMTP success, separate from concurrently updated run state."""
    run_id = os.getenv('LEARNDL_MANAGED_RUN')
    if not run_id:
        return
    store = RunStore()
    run = store.get(run_id)
    if run.get('owner') != 'idle_worklist' or run.get('pid') != os.getpid():
        return
    with store.connect() as connection:
        connection.execute('INSERT OR REPLACE INTO script_mail VALUES (?, ?, ?)',
                           (run_id, 'success' if success else 'error', time.time()))


def idle_exit_needs_email(store: RunStore, run: dict) -> bool:
    """Keep legacy no-script-mail runs covered across a rolling code update."""
    if run['phase'] == 'deferred' or run.get('no_email'):
        return False
    if run.get('task_kind') in {'rl_experiment', 'nn_metrics_ab', 'factor_sequence'}:
        from .recovery import evidence
        if (evidence(store, run['id']).get('script_notification_handoff')
                or run['phase'] == 'complete' or run.get('returncode') == 0):
            return False
        with store.connect() as connection:
            return connection.execute('SELECT 1 FROM script_mail WHERE run_id=?', (run['id'],)).fetchone() is None
    # Signals, reboot, launcher failure and watchdog termination are distinct
    # from an ordinary Python failure which the script may already have mailed.
    code = run.get('returncode')
    if code is None and 'term_at' not in run:
        from .recovery import exit_reason
        if exit_reason(store, run) == 'maintenance_pause':
            return False
    if 'term_at' in run or code is None or code < 0:
        return True
    if code == 75:
        return False
    if code == 0 or run['phase'] == 'complete':
        if run.get('script_email_enabled'):
            return False
        # Old workers launched training with email=False. Missing metadata must
        # therefore retain their watchdog result mail unless the script sent it.
        with store.connect() as connection:
            receipt = connection.execute("SELECT 1 FROM script_mail WHERE run_id=? AND outcome='success'",
                                         (run['id'],)).fetchone()
        return receipt is None
    with store.connect() as connection:
        receipt = connection.execute("SELECT 1 FROM script_mail WHERE run_id=? AND outcome='error'",
                                     (run['id'],)).fetchone()
    return receipt is None


def register_task(task_id: str) -> None:
    run_id = os.getenv('LEARNDL_MANAGED_RUN')
    if not run_id:
        return
    store = RunStore()
    run = store.get(run_id)
    if run and os.getsid(0) == run['pid']:
        store.link(task_id, run_id)


def timeout_override(task_id: str) -> dict[str, Any]:
    """A late recorder exit cannot overwrite a confirmed timeout outcome."""
    path = runtime_dir() / 'runs.sqlite'
    if not path.exists():
        return {}
    with sqlite3.connect(f'file:{path}?mode=ro', uri=True, timeout=15) as connection:
        row = connection.execute('SELECT data FROM runs JOIN task_links ON runs.id=task_links.run_id WHERE task_id=?', (task_id,)).fetchone()
    if row:
        run = json.loads(row[0])
        if run['phase'] in {'killed', 'finalizing'}:
            return {'status': 'killed', 'end_time': run['finished_at'], 'exit_code': -9 if run.get('kill_at') else -15,
                    'exit_error': timeout_message(run)}
    return {}


def process_cgroup(pid: int) -> str | None:
    try:
        for line in Path(f'/proc/{pid}/cgroup').read_text().splitlines():
            hierarchy, controllers, path = line.split(':', 2)
            if hierarchy == '0' or controllers == 'name=systemd':
                return path
    except FileNotFoundError:
        pass
    return None


def termination_requested() -> bool:
    if os.environ.get('LEARNDL_AUTOFIX_ID'):
        from .maintenance_gate import sandbox_request
        request = sandbox_request()
        if not isinstance(request, dict):
            raise RuntimeError('Invalid autofix execution context')
        marker = Path(request['channel']) / 'terminate.json'
        if marker.exists():
            return json.loads(marker.read_text()).get('nonce') == request['nonce']
        return False
    run_id = os.getenv('LEARNDL_MANAGED_RUN')
    return bool(run_id and 'term_at' in RunStore().get(run_id))


def in_run_scope(pid: int, run: dict) -> bool:
    if not run.get('cgroup'):
        return os.getsid(pid) == run['pid']
    group = process_cgroup(pid)
    return group is not None and (group == run['cgroup'] or group.startswith(run['cgroup'] + '/'))


def boot_id() -> str | None:
    try:
        return Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    except FileNotFoundError:
        return None


def previous_boot(run: dict) -> bool:
    return bool(run.get('boot_id') and run['boot_id'] != boot_id())


def members(run: dict) -> list[psutil.Process]:
    """Verify root identity and discover members of the dedicated session.

    Never use an unverified/reused root PID as a signal target. Descendants in
    the original session can remain after the root exits; PID creation times
    are checked again by psutil before each signal.
    """
    if previous_boot(run):
        return []
    try:
        root = psutil.Process(run['pid'])
        if abs(root.create_time() - run['process_created']) > .01:
            raise RuntimeError('Root PID has been reused; refusing termination')
    except psutil.NoSuchProcess:
        pass
    result = []
    for proc in psutil.process_iter(['pid', 'create_time', 'status', 'uids']):
        try:
            if not in_run_scope(proc.pid, run) or proc.pid == run.get('runner_pid') or proc.status() == psutil.STATUS_ZOMBIE:
                continue
            if proc.uids().real != run['uid'] or proc.create_time() < run['process_created'] - .01:
                raise RuntimeError('Session identity mismatch')
            result.append(proc)
        except (psutil.NoSuchProcess, ProcessLookupError):
            continue
    return result


def signal_members(run: dict, processes: list[psutil.Process], sig: signal.Signals) -> None:
    # Per-process signalling keeps PID-reuse checks inside psutil. Unlike
    # killpg, a newly reused numeric group is never blindly signalled.
    for proc in reversed(processes):
        try:
            if not in_run_scope(proc.pid, run) or proc.pid == run.get('runner_pid'):
                continue
            proc.send_signal(sig)
        except (psutil.NoSuchProcess, ProcessLookupError):
            continue


def timeout_message(run: dict) -> str:
    if run.get('operator_stop'):
        return f"Stopped by {run['operator_stop']}; automatic recovery disabled"
    if run.get('installation_stop'):
        return f"Stopped for installation {run['installation_stop']}; automatic recovery disabled"
    limit = run.get('progress_timeout_seconds') or run.get('timeout_seconds')
    condition = 'without output' if run.get('progress_timeout_seconds') else 'elapsed'
    return f'TIMEOUT: exceeded {limit} seconds {condition}; run {run["id"]}'


def timeout_due(run: dict, now: float) -> bool:
    if 'term_at' in run:
        return True
    if limit := run.get('progress_timeout_seconds'):
        last = run['started_at']
        if run.get('log_path'):
            try:
                last = max(last, Path(run['log_path']).stat().st_mtime)
            except FileNotFoundError:
                pass
        return now - last >= limit
    limit = run.get('timeout_seconds')
    return limit is not None and now - run['started_at'] >= limit


def finish_idle_event(store: RunStore, run: dict, detail: str) -> None:
    if run.get('owner') != 'idle_worklist' or run.get('operator_stop'):
        return
    if run.get('returncode') == 75 and 'term_at' not in run:
        run['phase'] = 'deferred'
        store.put(run)
    if not idle_exit_needs_email(store, run):
        return
    # Preserve the outcome context even if a later manual run replaces worklist state.
    try:
        from src.res.model.workflow.worklist_state import WorklistState
        state = WorklistState(run['task']).read() if run.get('training_priority') != 'promotion' and run.get('task_kind') not in {
            'rl_experiment', 'nn_metrics_ab', 'factor_sequence', 'model_promotion'
        } else None
        if state and state.get('revisions') == run.get('revisions'):
            run['completion'] = state
    except (OSError, ValueError):
        pass
    store.event(run, 'finished', detail)
    if not run.get('script_email_enabled'):
        # Recover legacy completion mail suppressed by the brief prior policy;
        # sent=1 is never reset, so delivered notifications remain deduplicated.
        with store.connect() as connection:
            connection.execute('UPDATE events SET sent=0 WHERE id=? AND sent=-1',
                               (f'{run["id"]}:finished',))


def _finish_production_event(store: RunStore, run: dict, detail: str) -> None:
    if (run.get('task') == 'production_model_update' and not run.get('operator_stop')
            and run.get('returncode') not in (0, 75)):
        # The script writes the same event for a completed batch, even on a
        # model error. INSERT OR IGNORE preserves that report if it exists.
        store.event(run, 'production_finished', detail)


def inspect_timeouts(store: RunStore | None = None, *, now: float | None = None) -> dict:
    store = store or RunStore()
    now = time.time() if now is None else now
    checked = 0
    for run in store.active():
        if run['phase'] == 'starting':
            # No child exists until runner has persisted its identity. A
            # crashed launcher before that point cannot be safely signalled.
            if now - run['started_at'] > 60:
                run.update(phase='error', finished_at=now)
                _finish_production_event(store, run, 'Launcher exited before registering a child.')
                store.put(run)
                finish_idle_event(store, run, 'Launcher exited before registering a child.')
            continue
        checked += 1
        if run.get('owner') == 'idle_worklist':
            store.event(run, 'started', 'Automatic schedule process started.')
        try:
            processes = members(run)
            if not processes:
                if 'returncode' not in run and run.get('runner_pid') and not previous_boot(run):
                    try:
                        runner = psutil.Process(run['runner_pid'])
                        if abs(runner.create_time() - run['runner_created']) < .01 and runner.status() != psutil.STATUS_ZOMBIE:
                            continue  # Let the live runner persist the actual exit code.
                    except psutil.NoSuchProcess:
                        pass
                timed_out = 'term_at' in run
                run.update(phase='finalizing' if timed_out else ('error' if run.get('returncode', 1) else 'complete'), finished_at=run.get('finished_at', now))
                detail = timeout_message(run) if timed_out else (
                    f'Process exited without a batch summary. Return code: {run.get("returncode", "unavailable")}.')
                _finish_production_event(store, run, detail)
                store.put(run)
                if timed_out:
                    from src.api.util.backend.task import TaskDatabase
                    database = TaskDatabase()
                    operator_logs = run.get('operator_crash_pins', {}).copy()
                    for task_id in store.links(run['id']):
                        task = database.get_task(task_id)
                        if task is None or (task.pid != run['pid'] and not task.is_running):
                            continue
                        updates = {
                            'status': 'killed', 'end_time': run['finished_at'], 'exit_code': -9 if run.get('kill_at') else -15,
                            'exit_error': timeout_message(run),
                        }
                        if run.get('operator_stop'):
                            paths = list(dict.fromkeys([*operator_logs.get(task_id, []), *pin_crash_logs(store, task)]))
                            operator_logs[task_id] = paths
                            updates['exit_files'] = list(dict.fromkeys([*paths, *(task.exit_files or [])]))
                        database.update_task(task_id, backend_updated=True, **updates)
                    run['phase'] = 'killed'
                    if run.get('operator_stop'):
                        run['operator_crash_logs'] = operator_logs
                        store.event(run, OPERATOR_STOP_EVENT, timeout_message(run))
                    elif run.get('owner') != 'idle_worklist' and run.get('task') != 'production_model_update':
                        store.event(run, 'terminated', timeout_message(run)+'; verified session processes have exited.')
                    store.put(run)
                finish_idle_event(store, run, f'Process finished: {run["phase"]}; returncode={run.get("returncode", "unavailable (abnormal exit)")}.')
                continue
            if not timeout_due(run, now):
                continue
            if 'term_at' not in run:
                run.update(phase='terminating', term_at=now)
                store.put(run)  # Persist intent before sending any signal.
            if now - run['term_at'] < 30:
                signal_members(run, processes, signal.SIGTERM)
            else:
                run['kill_at'] = run.get('kill_at', now)
                store.put(run)
                signal_members(run, processes, signal.SIGKILL)
                if now - run['kill_at'] >= 60:
                    store.event(run, 'failed', 'SIGKILL sent but processes are still present; will retry.')
        except (psutil.Error, OSError, RuntimeError) as exc:
            if timeout_due(run, now):
                store.event(run, 'failed', f'Timeout termination could not be verified/completed: {exc}')
    # Repair event persistence after a crash between storing state and inserting mail.
    for run in store.owned('idle_worklist'):
        if run.get('pid'):
            store.event(dict(run, phase='running'), 'started', 'Automatic schedule process started.')
        if run['phase'] in {'complete', 'error', 'killed', 'deferred'} and run.get('started_at'):
            finish_idle_event(store, run, f'Process finished: {run["phase"]}; returncode={run.get("returncode", "unavailable (abnormal exit)")}.')
    return {'checked_runs': checked}


def _recover_operator_task_events(store: RunStore) -> None:
    """Recover an interrupted Monitor request after its bounded signal wait."""
    with store.connect() as connection:
        rows = connection.execute('SELECT body FROM events WHERE sent=?', (_AWAITING_OPERATOR_EXIT,)).fetchall()
    now = time.time()
    for (body,) in rows:
        run = json.loads(body)['run']
        if now - run['stop_requested_at'] < 15:
            continue
        try:
            process = psutil.Process(run['pid'])
            exited = process.create_time() != run['process_created'] or process.status() == psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            exited = True
        except psutil.AccessDenied:
            continue
        store.finish_operator_task(run['id'], now if exited else None)


def _send_operator_stop(sender: Any, event_id: str, event: dict, mail_options: dict) -> bool:
    run = event['run']
    attachments = list(dict.fromkeys(Path(path) for paths in run['operator_crash_logs'].values()
                                     for path in paths if Path(path).is_file()))
    ended = datetime.fromtimestamp(run['finished_at'], timezone.utc).isoformat(timespec='seconds')
    message = (f'Task: {run["task"]}\nPID: {run["pid"]}\n'
               f'Termination confirmed at: {ended}\n{event["detail"]}\n'
               + ('Crash Protector output is attached.' if attachments else 'No Crash Protector output was available.'))
    return sender(f'Task Center - {run["task"]} - Stopped by operator', message,
                  attachments=attachments, confirmation_message='Learndl operator stop alert',
                  queue_on_failure=False, delivery_id=f'watchdog-event:{event_id}', **mail_options)


def deliver_timeout_events(sender: Any, store: RunStore | None = None) -> bool:
    store = store or RunStore()
    _recover_operator_task_events(store)
    with store.connect() as connection:
        rows = connection.execute('SELECT id,body FROM events WHERE sent=0').fetchall()
    success = True
    for event_id, body in rows:
        event = json.loads(body)
        run = event['run']
        promotion = run.get('task_kind') == 'model_promotion' or run.get('training_priority') == 'promotion'
        mail_options = ({'recipient': run.get('recipient') or run.get('config', {}).get('recipient')}
                        if run.get('task_kind') == 'rl_experiment' else {})
        if run.get('no_email'):
            with store.connect() as connection:
                connection.execute('UPDATE events SET sent=-1 WHERE id=?', (event_id,))
            continue
        if event['kind'] == OPERATOR_STOP_EVENT:
            try:
                sent = _send_operator_stop(sender, event_id, event, mail_options)
            except Exception:
                sent = False
            if sent:
                with store.connect() as connection:
                    connection.execute('UPDATE events SET sent=1 WHERE id=?', (event_id,))
            else:
                success = False
            continue
        if run.get('owner') == 'idle_worklist' and event['kind'] == 'configuration_error':
            # Missing schedules have no process/task record or result email.
            try:
                label = 'Model Lifecycle' if promotion else 'Idle Worklist'
                sent = sender(f'Watchdog {label} - configuration error', event['detail'],
                              confirmation_message='Learndl scheduler configuration alert', queue_on_failure=False,
                              delivery_id=f'watchdog-event:{event_id}',
                              **mail_options)
            except Exception:
                sent = False
            if sent:
                with store.connect() as connection:
                    connection.execute('UPDATE events SET sent=1 WHERE id=?', (event_id,))
            else:
                success = False
            continue
        if run.get('owner') == 'idle_worklist' and event['kind'] == 'finished':
            # Suppress new-policy success mails and errors reported since
            # this event was created; retain the event for audit (sent=-1).
            current = store.get(run['id']) or run
            if not idle_exit_needs_email(store, current):
                with store.connect() as connection:
                    connection.execute('UPDATE events SET sent=-1 WHERE id=?', (event_id,))
                continue
        from src.api.util.backend.task import TaskDatabase
        database = TaskDatabase()
        attachments = []
        for task_id in store.links(run['id']):
            task = database.get_task(task_id)
            if task:
                attachments.extend(Path(path) for path in task.exit_files or [] if Path(path).is_file())
                attachments.extend(Path(path) for path in task.get_crash_protector() if Path(path).is_file())
        production_batch = event['kind'].startswith('production_')
        label = 'Idle Worklist' if run.get('owner') == 'idle_worklist' else 'Timeout'
        if promotion:
            label = 'Model Lifecycle'
        message = (f'Task: {run["task"]}\nRun: {run["id"]}\nStarted: {run["started_at"]}\n'
                   f'Limit: {run["timeout_seconds"]} seconds\nElapsed: {time.time() - run["started_at"]:.0f} seconds\n'
                   f'Phase: {run["phase"]}\n{event["detail"]}\n'
                   f'Log: {run.get("log_path", "")}\nVersion: {run.get("version", "")}\n'
                   f'Progress timeout: {run.get("progress_timeout_seconds", "")} seconds')
        if production_batch:
            message = f'Run: {run["id"]}\n{event["detail"]}'
        if reference := run.get('schedule_reference'):
            message += f'\nSchedule reference: {reference}'
        if run.get('owner') == 'idle_worklist':
            from .recovery import evidence, recovery_plan
            from .idle_worklist import settings
            current = store.get(run['id']) or run
            if promotion:
                details = evidence(store, run['id'])
                if details.get('promotion_candidate'):
                    message += f'\nAccepted candidate: {details["promotion_candidate"]}'
                    if details.get('promotion_stage') == 'production':
                        message += '\nDeploy by explicitly registering this exact model in prediction.'
            if current.get('phase') not in {'queued', 'starting', 'running', 'deferred'}:
                message += '\nRecovery: ' + json.dumps(recovery_plan(store, current, settings()), ensure_ascii=False) + '\n'
                message += 'Exit evidence: ' + json.dumps(evidence(store, run['id']), ensure_ascii=False) + '\n'
            if run.get('parent_run_id'):
                message += f'\nAbnormal-exit resume attempt: {run["retry_index"]}; parent: {run["parent_run_id"]}; directory: selected by trainer\n'
            completion = run.get('completion') or {}
            message += (f'\nModel directory: {completion.get("model_path", "pending")}\n'
                        f'Training runs: {completion.get("training_run_ids", [])}\n'
                        f'Error: {completion.get("error", "")}\n'
                        f'Rerun mark: {run.get("rerun_mark", "none")}\n')
            for name, revision in run.get('revisions', {}).items():
                if isinstance(revision, dict):
                    message += f'{name} source: {revision.get("path", "")}\n'
                    message += f'{name} revision: {revision.get("sha256")} git: {revision.get("git_commit")}\n'
                else:
                    message += f'{name}: {json.dumps(revision, ensure_ascii=False)}\n'
        # Freeze dynamic elapsed/recovery text on the first attempt. A durable
        # delivery ID must refer to the same payload on every subsequent retry.
        if 'delivery_message' not in event:
            event['delivery_message'] = message
            with store.connect() as connection:
                connection.execute('UPDATE events SET body=? WHERE id=?', (json.dumps(event), event_id))
        message = event['delivery_message']
        title = (f'Production Model Update - {event["kind"].removeprefix("production_")}'
                 if production_batch else f'Watchdog {label} - {run["task"]} - {event["kind"]}')
        if sender(title, message,
                  attachments=list(dict.fromkeys(attachments)),
                  confirmation_message='Learndl production training' if production_batch else 'Learndl timeout alert',
                  queue_on_failure=False, delivery_id=f'watchdog-event:{event_id}', **mail_options):
            with store.connect() as connection:
                connection.execute('UPDATE events SET sent=1 WHERE id=?', (event_id,))
        else:
            success = False
    return success


def operator_task_stop_ids(store: RunStore | None = None) -> set[str]:
    """Ordinary tasks with an operator event owned by the watchdog mail queue."""
    if store is None:
        path = runtime_dir() / 'runs.sqlite'
        if not path.is_file():
            return set()
        store = RunStore(path)
    with store.connect() as connection:
        rows = connection.execute('SELECT id FROM events WHERE id LIKE ?', (f'{OPERATOR_TASK_PREFIX}%',)).fetchall()
    return {event_id.removeprefix(OPERATOR_TASK_PREFIX).removesuffix(f':{OPERATOR_STOP_EVENT}') for (event_id,) in rows}


def timeout_task_ids() -> set[str]:
    """Exclude tasks whose termination mail is owned by another watchdog event."""
    path = runtime_dir() / 'runs.sqlite'
    if not path.exists():
        return set()
    store = RunStore(path)
    with store.connect() as connection:
        rows = connection.execute('SELECT task_id,data FROM task_links JOIN runs ON runs.id=task_links.run_id').fetchall()
    return operator_task_stop_ids(store) | {
        task_id for task_id, data in rows
        if ('term_at' in (run := json.loads(data)) or run.get('owner') == 'idle_worklist'
            or run.get('task') == 'production_model_update')
    }
