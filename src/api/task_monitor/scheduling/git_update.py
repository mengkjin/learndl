"""Conservative, bounded master fast-forward for the short-lived watchdog."""
from __future__ import annotations

import json
import hashlib
import os
import signal
import subprocess
import time
from pathlib import Path
from datetime import datetime, timezone
from collections.abc import Callable

import portalocker
import psutil

from .config import runtime_dir
from .runtime import RunStore


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


def deliver_update_emails(sender: Callable, *, notice_id: str | None = None) -> bool:
    """Retry pending notices each watchdog tick; only mark delivered after SMTP success."""
    store = _mail_store()
    with store.connect() as connection:
        rows = connection.execute('SELECT id,data FROM git_update_mail WHERE sent=0' +
                                  (' AND id=?' if notice_id else ''), (notice_id,) if notice_id else ()).fetchall()
    delivered = True
    for key, raw in rows:
        data = json.loads(raw)
        files = data.get('changed_files', [])
        body = (f'Host: {data["host"]}\nRepository: {data["repository"]}\n'
                f'Branch: master (origin/master)\nUpdated at: {data["updated_at"]}\n'
                f'Before: {data["before"]}\nAfter: {data["after"]}\n'
                f'Changed files ({len(files)}):\n' + '\n'.join(files[:100]))
        if len(files) > 100:
            body += f'\n... {len(files) - 100} more; see watchdog state for the full list.'
        body += '\nFollow-up: ' + ('; '.join(data.get('follow_up', [])) or 'none detected')
        try:
            sent = sender(f'Watchdog Git Updated - {data["host"]} - {data["after"][:12]}', body,
                          confirmation_message='Learndl automatic Git update')
        except Exception:
            sent = False
        if sent:
            with store.connect() as connection:
                connection.execute('UPDATE git_update_mail SET sent=1 WHERE id=?', (key,))
        else:
            delivered = False
    return delivered


class GitUpdateError(RuntimeError):
    pass


def _git(root: Path, args: list[str], deadline: float, *, allow_failure: bool = False) -> str:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise GitUpdateError('Git update time budget exhausted')
    env = dict(os.environ, GIT_TERMINAL_PROMPT='0', GCM_INTERACTIVE='Never')
    # A non-interactive watchdog must never wait for an SSH password/passphrase.
    env.setdefault('GIT_SSH_COMMAND', 'ssh -o BatchMode=yes -o ConnectTimeout=10')
    command = ['git', '-c', 'core.hooksPath=/dev/null', '-c', 'submodule.recurse=false', '-C', str(root), *args]
    with subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True, start_new_session=True) as process:
        try:
            output, _ = process.communicate(timeout=remaining)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            raise GitUpdateError('Git command timed out') from None
        if process.returncode and not allow_failure:
            # Do not copy remote URLs/credential-helper diagnostics into watchdog state.
            raise GitUpdateError(f'git {args[0]} failed (exit {process.returncode}); inspect repository/authentication')
        return output.strip() if process.returncode == 0 else ''


def repository_blocker(root: Path, deadline: float) -> str | None:
    if _git(root, ['symbolic-ref', '--quiet', '--short', 'HEAD'], deadline, allow_failure=True) != 'master':
        return 'current branch is not master'
    for name in ('MERGE_HEAD', 'REBASE_HEAD', 'CHERRY_PICK_HEAD', 'REVERT_HEAD', 'rebase-merge', 'rebase-apply', 'sequencer', 'index.lock'):
        path = Path(_git(root, ['rev-parse', '--git-path', name], deadline))
        if (path if path.is_absolute() else root / path).exists():
            return 'another Git operation is in progress'
    if _git(root, ['status', '--porcelain=v1', '--untracked-files=normal', '--ignore-submodules=none'], deadline):
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
    try:
        with portalocker.Lock(str(directory / 'git_update.lock'), mode='a', timeout=0):
            return _update(root, options)
    except portalocker.LockException:
        return {'status': 'deferred', 'reason': 'another automatic update is in progress'}


def _update(root: Path, options: dict) -> dict:
    deadline = time.monotonic() + options.get('max_seconds', 45)
    result: dict = {}
    try:
        if reason := repository_blocker(root, deadline):
            return {'status': 'deferred', 'reason': reason}
        before = _git(root, ['rev-parse', 'HEAD'], deadline)
        result['before'] = before
        # Fetch only master. Keep the target hash fixed for all subsequent checks
        # instead of reading FETCH_HEAD which another fetch could overwrite.
        _git(root, ['fetch', '--no-tags', '--no-recurse-submodules', 'origin',
                    'refs/heads/master:refs/learndl/watchdog-master'], deadline)
        target = _git(root, ['rev-parse', 'refs/learndl/watchdog-master^{commit}'], deadline)
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
        result['checkout_attempted'] = True
        _git(root, ['-c', 'merge.autostash=false', 'merge', '--ff-only', '--no-edit', '--no-overwrite-ignore', target], deadline)
        after = _git(root, ['rev-parse', 'HEAD'], deadline)
        if after != target:
            raise GitUpdateError('HEAD changed unexpectedly during automatic update')
        follow_up = []
        if any(path in {'pyproject.toml', 'uv.lock', '.python-version'} or path.startswith('requirements') for path in changed):
            follow_up.append('dependency/runtime files changed; environment synchronization may be required')
        if any(path.startswith(('runs/scheduling/', 'runs/systemd/')) or path == 'src/api/task_monitor/scheduling/installer.py' for path in changed):
            follow_up.append('scheduling definitions changed; inspect plan/apply separately')
        result.update(status='updated', after=after, changed_files=changed, follow_up=follow_up)
        result['notification_id'] = queue_update_email(result, root)
        return result
    except (OSError, GitUpdateError) as exc:
        # Returning an error result still advances the five-minute poll deadline;
        # authentication/network failures must not cause a one-minute retry storm.
        return dict(result, status='error', reason=str(exc))
