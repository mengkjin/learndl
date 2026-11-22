"""Conservative, bounded master fast-forward for the short-lived watchdog."""
from __future__ import annotations

__all__ = [
    'queue_update_email',
    'queue_version_failure',
    'version_failure_mail_status',
    'deliver_update_emails',
    'GitUpdateError',
    'repository_blocker',
    'training_blocker',
    'check_and_update',
    'main',
]

import json
import hashlib
import os
import sqlite3
import signal
import subprocess
import time
import uuid
from pathlib import Path
from contextlib import closing
from datetime import datetime, timezone
from collections.abc import Callable
from typing import TYPE_CHECKING

import portalocker
import psutil

from .config import runtime_dir
from .runtime import RunStore

from src.automation.console import logger

if TYPE_CHECKING:
    from src.automation.version_patch import AttemptRecord


def _mail_store() -> RunStore:
    store = RunStore(runtime_dir() / 'runs.sqlite')
    with store.connect() as connection:
        connection.execute('CREATE TABLE IF NOT EXISTS git_update_mail '
                           '(id TEXT PRIMARY KEY, data TEXT NOT NULL, sent INTEGER NOT NULL DEFAULT 0)')
    return store


def queue_update_email(result: dict, root: Path) -> str:
    """Persist one success notice per repository/commit transition before sending."""
    from src.proj import MACHINE
    key = json.dumps([str(root.resolve()), result['before'], result['after']])
    notice_id = hashlib.sha256(key.encode()).hexdigest()
    data = dict(result, repository=str(root.resolve()), host=MACHINE.name,
                updated_at=datetime.now(timezone.utc).isoformat())
    store = _mail_store()
    with store.connect() as connection:
        connection.execute('INSERT OR IGNORE INTO git_update_mail(id,data) VALUES (?,?)',
                           (notice_id, json.dumps(data)))
    return notice_id


def queue_version_failure(attempt: AttemptRecord) -> None:
    """Queue one durable notification per failed or interrupted attempt."""
    data = {key: attempt[key] for key in
            ('patch', 'commit_hash', 'host', 'attempt', 'status', 'log_path')}
    from src.automation.patch_diagnostics import failure_summary

    data['kind'] = 'version_patch_failure'
    data['failure'] = failure_summary(attempt)
    with _mail_store().connect() as connection:
        connection.execute('INSERT OR IGNORE INTO git_update_mail(id,data) VALUES (?,?)',
                           ('version-' + attempt['id'], json.dumps(data)))


def version_failure_mail_status(attempt_id: str) -> str:
    """Read the failure notice status without creating the mail store."""
    path = runtime_dir() / 'runs.sqlite'
    try:
        with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)) as connection:
            row = connection.execute('SELECT sent,data FROM git_update_mail WHERE id=?',
                                     ('version-' + attempt_id,)).fetchone()
    except (OSError, sqlite3.Error):
        return 'unavailable'
    if row is None:
        return 'not queued'
    if row[0]:
        return 'SMTP accepted'
    return 'waiting for retry' if json.loads(row[1]).get('delivery_attempted') else 'queued'


def _record_repository_status(root: Path, status: str) -> None:
    """Atomically enqueue once per observed dirty episode, independent of file churn."""
    from src.proj import MACHINE
    repository = str(root.resolve())
    store = _mail_store()
    with store.connect() as db:
        db.execute('CREATE TABLE IF NOT EXISTS git_repository_notice '
                   '(repository TEXT PRIMARY KEY, notice_id TEXT NOT NULL)')
        db.execute('BEGIN IMMEDIATE')
        # Nullable first_seen keeps already-notified legacy records suppressed.
        columns = {row[1] for row in db.execute('PRAGMA table_info(git_repository_notice)')}
        if 'first_seen' not in columns:
            db.execute('ALTER TABLE git_repository_notice ADD COLUMN first_seen REAL')
        if not status:
            db.execute('DELETE FROM git_repository_notice WHERE repository=?', (repository,))
            return
        now = time.time()
        db.execute("INSERT OR IGNORE INTO git_repository_notice(repository,notice_id,first_seen) VALUES (?,'',?)",
                   (repository, now))
        old_notice, first_seen = db.execute(
            'SELECT notice_id,first_seen FROM git_repository_notice WHERE repository=?', (repository,)).fetchone()
        from .git_block_notice import WAIT_SECONDS
        if old_notice or now - first_seen < WAIT_SECONDS:
            return
        notice_id = 'dirty-' + uuid.uuid4().hex
        lines = status.splitlines()
        data = dict(kind='dirty_repository', host=MACHINE.name, repository=repository,
                    observed_at=datetime.now(timezone.utc).isoformat(),
                    first_seen=datetime.fromtimestamp(first_seen, timezone.utc).isoformat(),
                    waited_seconds=int(now-first_seen), file_status=lines[:100], entry_count=len(lines))
        db.execute('INSERT INTO git_update_mail(id,data) VALUES (?,?)', (notice_id, json.dumps(data)))
        db.execute('UPDATE git_repository_notice SET notice_id=? WHERE repository=?', (notice_id, repository))


def _repository_notice_body(data: dict) -> str:
    return (f'Host: {data["host"]}\nRepository: {data["repository"]}\n'
            f'Operation: Automatic Git update (origin/master)\n'
            f'Evidence captured: {data["observed_at"]}\n'
            f'First observed blocked: {data.get("first_seen", "not recorded")}\n'
            f'Observed wait: {data.get("waited_seconds", "not recorded")} seconds\n'
            'Reason: Local changes or untracked files; automatic update is paused.\n'
            'Remote availability has not been checked at this stage.\n'
            f'Git status entries ({data["entry_count"]}; first 100 shown):\n' +
            '\n'.join(data['file_status']) +
            '\nStatus: ?? = untracked; first column = index, second = working tree.\n'
            'Directories may summarize multiple untracked files. File contents are not included.\n'
            'Action: Review git status --short. Commit intended source changes, or move generated '
            'logs outside the checkout / configure appropriate ignore rules. Ignore rules do not '
            'hide changes to already tracked files. No files were deleted, stashed or overwritten.\n'
            'Checks continue normally and updates resume when all safety checks pass. '
            'This episode is notified once, even if files keep changing. Failed delivery retries '
            'this same snapshot; it may no longer reflect the state at delivery time.')


def _queue_completed_patch_mail(store: RunStore) -> bool:
    from src.automation.version_patch import MigrationError, queue_completed_batches

    def enqueue(batch_id: str, data: str) -> None:
        with store.connect() as connection:
            connection.execute('INSERT OR IGNORE INTO git_update_mail(id,data) VALUES (?,?)',
                               ('version-success-' + batch_id, data))

    try:
        queue_completed_batches(runtime_dir().parent, enqueue)
    except (OSError, sqlite3.Error, MigrationError) as exc:
        logger.error('Could not queue completed version patch emails: %s', exc)
        return False
    return True


def deliver_update_emails(sender: Callable, *, notice_id: str | None = None) -> bool:
    """Retry pending notices each watchdog tick; only mark delivered after SMTP success."""
    store = _mail_store()
    delivered = notice_id is not None or _queue_completed_patch_mail(store)
    with store.connect() as connection:
        rows = connection.execute('SELECT id,data FROM git_update_mail WHERE sent=0' +
                                  (' AND id=?' if notice_id else ''), (notice_id,) if notice_id else ()).fetchall()
    for key, raw in rows:
        data = json.loads(raw)
        if data.get('kind') == 'version_patch_failure':
            subject = f'Learndl version patch failed - {data["host"]} - {data["patch"]}'
            from src.automation.patch_diagnostics import failure_summary

            failure = data.get('failure') or failure_summary(data)
            body = (f'Patch: {data["patch"]}\nCommit: {data["commit_hash"]}\n'
                    f'Attempt: {data["attempt"]} ({data["status"]})\n'
                    f'Step: {failure.get("step", "run")}\nError: {failure.get("error", "See the patch log")}\n')
            for field in ('source', 'target', 'item', 'blocked_count', 'report', 'backup_dir'):
                if field in failure:
                    body += f'{field}: {failure[field]}\n'
            body += (f'Log: {data["log_path"]}\nNew business tasks are paused.\n'
                     f'Retry: {failure.get("retry") or (".venv/bin/python -m src.automation.version_patch retry " + data["patch"])}')
        elif data.get('kind') == 'version_patch_success':
            patches = '\n'.join(f'- {patch_id}' for patch_id in data['patches'])
            finished = datetime.fromtimestamp(data['finished_at'], timezone.utc).isoformat()
            subject = f'Learndl version patches completed - {data["host"]} - {data["target_version"]}'
            body = (f'Host: {data["host"]}\nCommit: {data["commit_hash"]}\n'
                    f'Data version: {data["source_version"]} -> {data["target_version"]}\n'
                    f'Completed at: {finished}\nVerified patches ({len(data["patches"])}):\n{patches}\n'
                    f'Attempt logs: {data["attempts_dir"]}')
        elif data.get('kind') == 'git_operation':
            subject = f'Watchdog Git Update Blocked - {data["host"]} - Git operation or lock'
            body = data['body']
        elif data.get('kind') == 'dirty_repository':
            subject = f'Watchdog Git Update Paused - {data["host"]} - Local changes'
            body = _repository_notice_body(data)
        else:
            files = data.get('changed_files', [])
            body = (f'Host: {data["host"]}\nRepository: {data["repository"]}\n'
                    f'Branch: master (origin/master)\nUpdated at: {data["updated_at"]}\n'
                    f'Before: {data["before"]}\nAfter: {data["after"]}\n'
                    f'Changed files ({len(files)}):\n' + '\n'.join(files[:100]))
            if len(files) > 100:
                body += f'\n... {len(files) - 100} more; see watchdog state for the full list.'
            body += '\nFollow-up: ' + ('; '.join(data.get('follow_up', [])) or 'none detected')
            subject = f'Watchdog Git Updated - {data["host"]} - {data["after"][:12]}'
        try:
            sent = sender(subject, body,
                          confirmation_message='Learndl automatic Git update', queue_on_failure=False,
                          delivery_id=f'git-update:{key}')
        except Exception:
            sent = False
        if sent:
            with store.connect() as connection:
                connection.execute('UPDATE git_update_mail SET sent=1 WHERE id=?', (key,))
        else:
            if data.get('kind') == 'version_patch_failure' and not data.get('delivery_attempted'):
                data['delivery_attempted'] = True
                with store.connect() as connection:
                    connection.execute('UPDATE git_update_mail SET data=? WHERE id=? AND sent=0',
                                       (json.dumps(data), key))
            delivered = False
    return delivered


class GitUpdateError(RuntimeError):
    pass


def _timeout_audit(root, args, pid, termination):
    # Never persist remote URLs, credentials, command output or arbitrary arguments.
    temporary = runtime_dir() / f'.git-timeout-{uuid.uuid4().hex}.tmp'
    try:
        temporary.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(dict(
            repository=str(root.resolve()), command=args[0], pid=pid,
            observed_at=datetime.now(timezone.utc).isoformat(), termination=termination)))
        temporary.replace(runtime_dir() / 'git-timeout.json')
    except OSError:
        pass  # Diagnostics must not prevent terminating a timed-out subprocess.
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _git(root: Path, args: list[str], deadline: float, *, allow_failure: bool = False) -> str:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise GitUpdateError('Git update time budget exhausted')
    env = dict(os.environ, GIT_TERMINAL_PROMPT='0', GCM_INTERACTIVE='Never',
               GIT_OPTIONAL_LOCKS='0')
    # A non-interactive watchdog must never wait for an SSH password/passphrase.
    env.setdefault('GIT_SSH_COMMAND', 'ssh -o BatchMode=yes -o ConnectTimeout=10')
    command = ['git', '-c', 'core.hooksPath=/dev/null', '-c', 'submodule.recurse=false', '-C', str(root), *args]
    with subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True, start_new_session=True) as process:
        try:
            output, _ = process.communicate(timeout=remaining)
        except subprocess.TimeoutExpired:
            _timeout_audit(root, args, process.pid, 'TERM requested')
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                _timeout_audit(root, args, process.pid, 'KILL required after 5-second TERM grace')
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.communicate(timeout=5)
            else:
                _timeout_audit(root, args, process.pid, 'Exited during TERM grace; no KILL sent')
            raise GitUpdateError('Git command timed out; inspect git-timeout.json for termination evidence') from None
        if process.returncode and not allow_failure:
            # Do not copy remote URLs/credential-helper diagnostics into watchdog state.
            raise GitUpdateError(f'git {args[0]} failed (exit {process.returncode}); inspect repository/authentication')
        if process.returncode:
            return ''
        return output.rstrip('\n') if '--porcelain=v1' in args else output.strip()


def repository_blocker(root: Path, deadline: float) -> str | None:
    if _git(root, ['symbolic-ref', '--quiet', '--short', 'HEAD'], deadline, allow_failure=True) != 'master':
        return 'current branch is not master'
    from .git_block_notice import MARKERS, observe
    markers = []
    for name in MARKERS:
        path = Path(_git(root, ['rev-parse', '--git-path', name], deadline))
        path = path if path.is_absolute() else root / path
        try:
            info = path.stat()
        except FileNotFoundError:
            continue
        markers.append(dict(path=str(path), mtime=info.st_mtime, size=info.st_size))
    observe(_mail_store(), root, markers)
    if markers:
        return 'another Git operation is in progress: ' + ', '.join(item['path'] for item in markers)
    status = _git(root, ['-c', 'core.quotePath=true', 'status', '--porcelain=v1', '--untracked-files=normal', '--ignore-submodules=none'], deadline)
    _record_repository_status(root, status)
    if status:
        return 'local changes or untracked files'
    return None


def training_blocker() -> str | None:
    """FitLock plus full lifecycle records cover data loading and final evaluation."""
    from src.proj import PATH
    from src.proj.util.script import FitLock
    if FitLock.is_held():
        return 'fit lock occupied'
    if RunStore().active():
        return 'managed task is active'
    for path in (PATH.lc_machine / 'training_history').glob('*/run.json'):
        try:
            run = json.loads(path.read_text())
            if run['status'] != 'running':
                continue
            process = psutil.Process(run['pid'])
            if abs(process.create_time() - run['process_created']) < .01 and process.status() != psutil.STATUS_ZOMBIE:
                return 'training/evaluation pipeline is active'
        except psutil.NoSuchProcess:
            continue
        except (OSError, psutil.Error, ValueError, KeyError, TypeError):
            return 'training activity cannot be verified'
    return None


def check_and_update(options: dict, *, root: Path | None = None) -> dict:
    from src.proj import MACHINE, PATH
    root = root or PATH.main
    if not MACHINE.platform_server:
        return {'status': 'skipped', 'reason': 'server only'}
    directory = runtime_dir()
    directory.mkdir(parents=True, exist_ok=True)
    from .maintenance_gate import MaintenanceBusy, permit
    from . import channels
    if channels.enabled():
        return _channel_update(root, options)
    try:
        with permit(exclusive=True, blocking=False, task_class='git'):
            from . import priority
            if priority.enabled():
                from .autofix_store import Store, state_dir
                controls = Store(state_dir())
                current = priority.read(controls)
                if current['phase'] not in {'checkout', 'blocked'}:
                    priority.write(controls, dict(current, phase='draining'))
            with portalocker.Lock(str(directory / 'git_update.lock'), mode='a', timeout=0):
                return _admitted_update(root, options)
    except (portalocker.LockException, MaintenanceBusy):
        return {'status': 'deferred', 'reason': 'execution/code lease occupied; Git retains priority'}


def _admitted_update(root, options, *, target_snapshot=None, migrate=None):
    from . import priority
    if not priority.enabled():
        return _update(root, options)
    from .autofix_store import Store, state_dir
    store = Store(state_dir())
    state = priority.read(store)
    if state['phase'] in {'checkout', 'blocked'} and target_snapshot is None:
        # This is a fresh process after an attempted checkout. Never
        # import/dispatch work from the process that changed the tree.
        deadline = time.monotonic() + options.get('max_seconds', 45)
        try:
            reason = repository_blocker(root, deadline)
            head = _git(root, ['rev-parse', 'HEAD'], deadline)
            if reason or head != state.get('target'):
                raise GitUpdateError(reason or 'checkout HEAD differs from the intended target')
        except (OSError, GitUpdateError) as exc:
            priority.write(store, dict(state, phase='blocked', reason=str(exc), version_verified=False))
            return {'status': 'blocked', 'reason': str(exc)}
        if migrate is not None:
            # The mail worker can now load the verified checkout even if a
            # migration fails. Business workers still see checkout/blocked.
            state = dict(state, version_verified=True, version_verified_head=head)
            priority.write(store, state)
            result = migrate()
            if result['status'] != 'ready':
                priority.write(store, dict(state, phase='blocked', reason=result['reason']))
                return result
        priority.write(store, {'phase': 'idle', 'last_completed': time.time(), 'verified_head': head})
        return {'status': 'verified', 'after': head}
    if options.get('enabled') is False:
        priority.write(store, {'phase': 'idle', 'last_completed': time.time()})
        return {'status': 'disabled'}
    try:
        result = _update(root, options) if target_snapshot is None else _update(root, options, target_snapshot=target_snapshot)
    except Exception as exc:
        # Even a notice/database error after merge must end this Python process.
        result = {'status': 'error', 'reason': f'{type(exc).__name__}: {exc}',
                  'checkout_attempted': priority.read(store)['phase'] == 'checkout'}
    if not result.get('checkout_attempted') and state['phase'] != 'blocked':
        priority.write(store, {'phase': 'idle', 'last_completed': time.time(), 'result': result})
    return result

def _update(root: Path, options: dict, *, target_snapshot=None) -> dict:
    deadline = time.monotonic() + options.get('max_seconds', 45)
    result: dict = {}
    try:
        if reason := repository_blocker(root, deadline):
            return {'status': 'deferred', 'reason': reason}
        before = _git(root, ['rev-parse', 'HEAD'], deadline)
        result['before'] = before
        # Fetch only master. Keep the target hash fixed for all subsequent checks
        # instead of reading FETCH_HEAD which another fetch could overwrite.
        if target_snapshot is None:
            _git(root, ['fetch', '--no-tags', '--no-recurse-submodules', 'origin',
                        'refs/heads/master:refs/learndl/watchdog-master'], deadline)
            target = _git(root, ['rev-parse', 'refs/learndl/watchdog-master^{commit}'], deadline)
        else:
            expected_before, target = target_snapshot
            if before != expected_before:
                return dict(result, status='deferred', reason='HEAD changed since remote check')
        result = {'before': before, 'remote_head': target}
        if before == target:
            return dict(result, status='up_to_date')
        # Empty output signals merge-base exit 1; asking for the actual base also
        # distinguishes local-ahead/diverged histories without changing them.
        base = _git(root, ['merge-base', before, target], deadline, allow_failure=True)
        if base != before:
            return dict(result, status='deferred', reason='local commits or divergent master; manual review required')
        if reason := training_blocker():
            return dict(result, status='deferred', reason=reason)
        changed = _git(root, ['diff', '--name-only', before, target], deadline).splitlines()
        # Recheck after network access, immediately before replacing any files.
        if reason := repository_blocker(root, deadline):
            return dict(result, status='deferred', reason=reason)
        if _git(root, ['rev-parse', 'HEAD'], deadline) != before:
            return dict(result, status='deferred', reason='local HEAD changed during check')
        if reason := training_blocker():
            return dict(result, status='deferred', reason=reason)
        if deadline - time.monotonic() < 10:
            return dict(result, status='deferred', reason='insufficient remaining checkout time budget')
        from . import priority
        if priority.enabled():
            from .autofix_store import Store, state_dir
            store = Store(state_dir())
            # Intent precedes the first write to the checkout; crashes remain
            # blocked until a new process verifies the exact intended commit.
            priority.write(store, dict(priority.read(store), phase='checkout', target=target, before=before,
                                       version_verified=False))
        result['checkout_attempted'] = True
        _git(root, ['-c', 'merge.autostash=false', 'merge', '--ff-only', '--no-edit', '--no-overwrite-ignore', target], deadline)
        after = _git(root, ['rev-parse', 'HEAD'], deadline)
        if after != target:
            raise GitUpdateError('HEAD changed unexpectedly during automatic update')
        follow_up = []
        if any(path in {'pyproject.toml', 'uv.lock', '.python-version'} or path.startswith('requirements') for path in changed):
            follow_up.append('dependency/runtime files changed; environment synchronization may be required')
        installed_sources = {'src/automation/scheduling/' + name + '.py'
                             for name in ('installer', 'worker_bootstrap', 'autofix_sandbox', 'install_privileged')}
        if any(path.startswith(('configs/automation/', 'runs/scheduling/', 'runs/systemd/')) or
               path in installed_sources or path == 'src/automation/scheduling/installer.py' for path in changed):
            follow_up.append('scheduling definitions changed; inspect plan/apply separately')
        if any(path in {'configs/automation/autofix.yaml', 'runs/autofix/tasks.yaml'} or path.startswith('scripts/.autofix/') for path in changed):
            follow_up.append('autofix definitions changed; the separate worker validates new enabled IDs; '
                             'this Git notification does not confirm Autofix startup or completion')
        result.update(status='updated', after=after, changed_files=changed, follow_up=follow_up)
        result['notification_id'] = queue_update_email(result, root)
        return result
    except (OSError, GitUpdateError) as exc:
        # Returning an error result still advances the five-minute poll deadline;
        # authentication/network failures must not cause a one-minute retry storm.
        return dict(result, status='error', reason=str(exc))


def _automatic_patch_retry(root, runtime, deadline, head=None):
    """Select a failed patch only once per clean fast-forward commit."""
    from src.automation import version_patch

    snapshot = version_patch.status(root, runtime)
    failed = snapshot['failed']
    if failed is None:
        return None
    head = head or _git(root, ['rev-parse', 'HEAD'], deadline)
    if any(row['patch'] == failed['patch'] and row['commit_hash'] == head for row in snapshot['records']):
        return None
    if repository_blocker(root, deadline):
        return None
    previous = failed['commit_hash']
    if _git(root, ['merge-base', previous, head], deadline, allow_failure=True) == previous:
        return failed['patch']
    return None


def _channel_update(root, options):
    """Reconcile data, then fetch only when a fast-forward can be admitted."""
    from . import channels, priority
    from .autofix_store import Store, state_dir
    from .maintenance_gate import permit, MaintenanceBusy, pass_fds
    from .config import installed_config
    from src.automation import version_patch as version
    from src.automation.patch_services import PatchServices
    from src.shared.file_ops import atomic_json

    store = Store(state_dir())
    directory = runtime_dir()
    runtime = directory.parent
    result = {}
    keep_reservation = False
    with portalocker.Lock(str(directory / 'git-worker.lock'), timeout=0):
        try:
            state = priority.read(store)
            activated = version.activation(installed_config())
            retry_enabled = activated and bool(options.get('enabled')) and options.get('retry_failed_patch_on_update', True)
            checkout_pending = state['phase'] in {'checkout', 'blocked'}
            needs_reconciliation = checkout_pending or (
                activated and (version.needs_migration(root, runtime) or PatchServices(root, runtime).reconcile_needed())
            )

            def migrate(fd):
                if reason := training_blocker():
                    return {'status': 'deferred', 'reason': reason}
                try:
                    deadline = time.monotonic() + options.get('max_seconds', 45)
                    retry = None
                    # An empty ledger can still initialize through apply_pending.
                    if retry_enabled and version.ledger_path(runtime).is_file():
                        retry = _automatic_patch_retry(root, runtime, deadline)
                    return version.apply_pending(root, runtime, pass_fds=(fd, *pass_fds()),
                                                 notify=queue_version_failure, retry=retry)
                except Exception as exc:
                    return {'status': 'blocked', 'reason': f'{type(exc).__name__}: {exc}'}

            if needs_reconciliation:
                with permit(exclusive=True, blocking=False, task_class='git'), version.code_lease(directory) as fd:
                    if checkout_pending:
                        if retry_enabled and state.get('version_verified_head') == state.get('target'):
                            deadline = time.monotonic() + options.get('max_seconds', 45)
                            head = _git(root, ['rev-parse', 'HEAD'], deadline)
                            if head != state['target']:
                                if retry := _automatic_patch_retry(root, runtime, deadline, head):
                                    version.verify_manual_checkout(root, runtime, retry=retry)
                        result = _admitted_update(root, options, migrate=(lambda: migrate(fd)) if activated else None)
                    else:
                        result = migrate(fd)
                state = priority.read(store)
                if (not retry_enabled or result.get('status') != 'blocked' or not result.get('attempt_id')
                        or (checkout_pending and not state.get('version_verified'))):
                    return result

            if not options.get('enabled'):
                result = {'status': 'disabled'}
                return result
            deadline = time.monotonic() + options.get('max_seconds', 45)
            if reason := repository_blocker(root, deadline):
                result = dict(result, git_update={'status': 'deferred', 'reason': reason}) if result else {
                    'status': 'deferred', 'reason': reason}
                return result
            before = _git(root, ['rev-parse', 'HEAD'], deadline)
            _git(root, ['fetch', '--no-tags', '--no-recurse-submodules', 'origin',
                        'refs/heads/master:refs/learndl/watchdog-master'], deadline)
            target = _git(root, ['rev-parse', 'refs/learndl/watchdog-master^{commit}'], deadline)
            if target == before:
                result = result or {'status': 'up_to_date'}
                return result
            if _git(root, ['merge-base', before, target], deadline, allow_failure=True) != before:
                result = {'status': 'deferred', 'reason': 'local commits or divergent master'}
                return result
            keep_reservation = True
            if not channels.reserve(store, 'git', channels.GROUPS):
                result = {'status': 'deferred', 'reason': 'maintenance occupied or cooling down'}
                return result
            with permit(exclusive=True, blocking=False, task_class='git'), version.code_lease(directory):
                if reason := training_blocker():
                    result = {'status': 'deferred', 'reason': reason}
                    return result
                channels.running(store, 'git')
                result = _admitted_update(root, options, target_snapshot=(before, target))
                keep_reservation = False
                return result
        except (portalocker.LockException, MaintenanceBusy, version.CodeBusy) as exc:
            from .maintenance_notice import record_failure
            record_failure(store, 'git', exc)
            result = {'status': 'deferred', 'reason': str(exc) or 'code users still active'}
            return result
        except (OSError, GitUpdateError, version.MigrationError) as exc:
            result = dict(result, status='error', reason=str(exc))
            return result
        finally:
            if (not keep_reservation and not result.get('checkout_attempted')
                    and priority.read(store)['phase'] not in {'checkout', 'blocked'}):
                channels.release(store, 'git')
            atomic_json(directory / 'git-channel.json', dict(result, checked_at=time.time()))


def main():
    from .config import installed_config
    from . import channels
    from src.shared.file_ops import atomic_json
    config = installed_config() or {}
    if not channels.enabled():
        return 0
    options = config['watchdog']['jobs'].get('git_auto_update', {})
    try:
        result = check_and_update(options)
    except Exception as exc:
        result = {'status': 'error', 'reason': f'{type(exc).__name__}: {exc}'}
    # On checkout, do not import project modules after the tree changed.
    atomic_json(runtime_dir() / 'git-channel.json', dict(result, checked_at=time.time()))
    logger.info(json.dumps(result, ensure_ascii=False))
    return int(result.get('status') in {'error', 'blocked'})


if __name__ == '__main__':
    raise SystemExit(main())
