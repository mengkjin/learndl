"""At-most-once script requests. validate/plan/status do not import submitted scripts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import portalocker
import psutil

from .autofix_spec import ManifestError, canonical, manifest, redact, validate
from .autofix_store import ACTIVE, ROOT, Store, state_dir
from . import autofix_mailbox as mailbox


def commit(root):
    result = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=10)
    return result.stdout.strip() if result.returncode == 0 else 'unknown'


def reconcile(store):
    """Never rerun an uncertain claim. Keep admission closed while survivors exist."""
    active = False
    for row in store.rows():
        if row['phase'] not in ACTIVE:
            continue
        try:
            process = psutil.Process(row['supervisor_pid'])
            if abs(process.create_time() - row['supervisor_created']) < .01 and process.status() != psutil.STATUS_ZOMBIE:
                active = True
                continue
        except psutil.NoSuchProcess:
            pass
        except (psutil.Error, KeyError):
            active = True
            continue
        # A supervisor crash does not prove its children have stopped.
        if row.get('scope'):
            try:
                from .runtime import members
                if members(row['scope']):
                    active = True
                    continue
            except (psutil.Error, KeyError, RuntimeError, OSError):
                active = True
                continue
        if row.get('pid'):
            try:
                if row.get('cgroup'):
                    from .runtime import members
                    if members(row):
                        active = True
                        continue
                child = psutil.Process(row['pid'])
                if abs(child.create_time() - row['process_created']) < .01 and child.status() != psutil.STATUS_ZOMBIE:
                    active = True
                    continue
            except psutil.NoSuchProcess:
                pass
            except (psutil.Error, KeyError, RuntimeError, OSError):
                active = True
                continue
        store.update(row['id'], phase='unknown', event_kind='interrupted', finished_at=time.time(), error='Supervisor disappeared; execution may have occurred. This ID will never be retried.')
    return active


def discover(store, root):
    path = root / 'runs/autofix/tasks.yaml'
    try:
        items = manifest(path)
    except ManifestError as exc:
        fingerprint = hashlib.sha256(path.read_bytes() if path.exists() else b'missing').hexdigest()
        store.event('manifest:' + fingerprint + ':' + type(exc).__name__,
                    {'kind': 'configuration_error', 'host': socket.gethostname(), 'commit': commit(root),
                     'error': str(exc), 'file': str(path)})
        return False
    store.register(items, commit(root))
    return True


def supervise(store, now=None):
    """Watchdog backstop, using the scheduler's verified process-scope signalling."""
    from .runtime import members, signal_members
    now = time.time() if now is None else now
    for row in store.rows():
        limit = row.get('effective', {}).get('timeout_seconds')
        if row['phase'] not in ACTIVE or limit is None or now - row['started_at'] < limit:
            continue
        try:
            if 'term_at' not in row:
                row = store.update(row['id'], term_at=now)
            if row.get('channel'):
                try:
                    mailbox.write(Path(row['channel']), 'terminate.json', {'nonce': row['nonce']})
                except OSError:
                    pass  # A damaged mailbox must not prevent process termination.
            sig = signal.SIGTERM if now - row['term_at'] < 30 else signal.SIGKILL
            if row.get('pid'):
                signal_members(row, members(row), sig)
            else:
                process = psutil.Process(row['supervisor_pid'])
                if abs(process.create_time() - row['supervisor_created']) < .01:
                    process.send_signal(sig)
        except psutil.NoSuchProcess:
            pass
        except (OSError, psutil.Error, RuntimeError) as exc:
            store.event(row['id'] + ':supervision_failed', dict(row, kind='supervision_failed', error=str(exc)))


def tick():
    directory = state_dir()
    store = Store(directory)
    supervise(store)
    active = reconcile(store)
    valid = discover(store, ROOT)
    queued = [row for row in store.rows() if row['phase'] == 'queued'] if valid else []
    if queued or active:
        (directory / 'pending').touch()
    else:
        (directory / 'pending').unlink(missing_ok=True)
    return {'queued': len(queued), 'active': active, 'manifest_valid': valid,
            'state_directory': str(directory), 'scheduled_waiting': len(store.scheduled()),
            'outcomes': {phase: sum(r['phase'] == phase for r in store.rows()) for phase in ('success', 'failed', 'unknown')}}


def _mail_body(event):
    # Parameter values are excluded from exception/log text to avoid accidental leaks.
    item = event.get('item', {})
    if event.get('kind') == 'starting':
        claimed_at = datetime.fromtimestamp(event['started_at'], timezone.utc).isoformat()
        return '\n'.join([
            'Autofix 启动通知：已领取本 ID，开始启动流程。',
            '这是领取时的记录，不代表业务脚本已开始或已经成功；邮件可能因队列重试而延迟送达。',
            f'Autofix ID: {event["id"]}', f'Host: {event.get("host", socket.gethostname())}',
            f'Claimed at (UTC): {claimed_at}', f'Git commit: {event.get("commit", "unknown")}',
            f'Script: {item.get("script", "")}', f'Parameters: {canonical(redact(item.get("params", {})))}',
            f'Log: {event.get("log_path", "")}',
            '本通知由 watchdog 发送。Autofix 强制传入 email=true，脚本负责结果邮件。上方参数为原始登记声明。',
            '本 ID 最多尝试一次；失败后如需重新执行，必须新增不同 ID。'])
    lines = [f'Host: {event.get("host", socket.gethostname())}', f'Autofix ID: {event.get("id", "manifest")}',
             f'Git commit: {event.get("commit", "unknown")}', f'Script: {item.get("script", "")}',
             f'Parameters: {canonical(redact(item.get("params", {})))}',
             f'Phase: {event.get("phase", event.get("kind", ""))}',
             f'Process created: {bool(event.get("pid"))}',
             f'Script notification ownership: {bool(event.get("handoff"))}',
             f'Error: {event.get("error", "")}', f'Log: {event.get("log_path", "")}',
             'No script retry will occur. To retry a task, append a different ID.']
    if event.get('error_tail'):
        lines += ['Error output (bounded):', event['error_tail']]
    return '\n'.join(lines)


def deliver(sender):
    store = Store(state_dir())
    complete = True
    with portalocker.Lock(str(store.directory / 'mail.lock'), timeout=0):
        with store.connect(readonly=True) as db:
            events = db.execute('SELECT id,data FROM events WHERE sent=0 ORDER BY rowid').fetchall()
        for key, raw in events:
            event = json.loads(raw)
            try:
                sent = sender(f'Autofix {event["kind"]} - {event.get("id", "tasks.yaml")}', _mail_body(event),
                              confirmation_message='Learndl autofix alert', queue_on_failure=False)
            except Exception:
                sent = False
            if sent:
                with store.connect() as db:
                    db.execute('UPDATE events SET sent=1 WHERE id=?', (key,))
            else:
                complete = False
    return complete


def _read_event(channel, kind, nonce):
    try:
        value = mailbox.read(channel, f'{kind}.json')
        return value if value.get('nonce') == nonce and value.get('kind') == kind else None
    except (OSError, ValueError):
        return None


def _error_tail(path, params):
    if not path.exists():
        return ''
    with path.open('rb') as file:
        file.seek(max(0, path.stat().st_size - 6000))
        text = file.read(6000).decode(errors='replace')
    for key, value in params.items():
        if any(word in key.lower() for word in ('password', 'token', 'secret', 'credential', 'api_key')):
            text = text.replace(str(value), '[REDACTED]')
    return text


def execute(store, root, row):
    """Caller holds exclusive maintenance permission. Claim precedes validation/spawn."""
    key = row['id']
    from .runtime import process_cgroup, boot_id
    created = psutil.Process().create_time()
    scope = {'pid': os.getpid(), 'runner_pid': os.getpid(), 'process_created': created,
             'uid': os.getuid(), 'boot_id': boot_id(), 'cgroup': process_cgroup(os.getpid())}
    if not (scope['cgroup'] or '').endswith('/learndl-autofix.service'):
        scope = None
    log = store.directory / 'logs' / (hashlib.sha256(key.encode()).hexdigest() + '.log')
    record = store.claim(key, supervisor_created=created, commit=commit(root), scope=scope, log_path=str(log))
    if record is None:
        return
    process = None
    handoff = False
    timed_out = False
    try:
        import importlib.util
        module_spec = importlib.util.spec_from_file_location('learndl_installed_autofix_policy', store.directory / 'policy.py')
        policy = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(policy)
        check_tree, command = policy.check_tree, policy.command
        spec = validate(root, record['item'])
        store.update(key, effective=spec, log_path=str(log))
        deadline = time.monotonic() + max(0, spec['timeout_seconds'] - (time.time() - record['started_at']))
        check_tree(root.resolve())
        channel = root / '.local_resources/autofix_channels' / secrets.token_hex(16)
        mailbox.create(channel)
        nonce = secrets.token_hex(32)
        request = store.directory / 'request.json'
        request.write_text(canonical({'id': key, 'nonce': nonce, 'channel': str(channel), 'script': spec['script'], 'params': spec['params']}))
        argv = [sys.executable, '-u', '-m', 'src.api.task_monitor.scheduling.autofix', '_child',
                str(root / spec['script']), '--source=autofix', *spec['args']]
        invocation = command(root, request, argv)
        store.update(key, effective=spec, log_path=str(log), nonce=nonce, channel=str(channel))
        if time.monotonic() >= deadline:
            raise TimeoutError('Autofix startup exceeded the execution timeout')
        env = dict(os.environ, LEARNDL_AUTOFIX_ID=key, LEARNDL_AUTOFIX_NONCE=nonce, PYTHONUNBUFFERED='1')
        info_path = store.directory / 'sandbox-info.json'
        with log.open('ab', buffering=0) as output, info_path.open('w') as info:
            separator = invocation.index('--')
            invocation[separator:separator] = ['--info-fd', str(info.fileno())]
            process = subprocess.Popen(invocation, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                       stdout=output, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(info.fileno(),))
            store.update(key, pid=process.pid, process_created=psutil.Process(process.pid).create_time(), phase='running',
                         runner_pid=os.getpid(), uid=os.getuid(), boot_id=boot_id(), cgroup=scope['cgroup'] if scope else None)
            while process.poll() is None:
                if not (channel / 'host.json').exists():
                    try:
                        host_pid = json.loads(info_path.read_text())['child-pid']
                        mailbox.write(channel, 'host.json', {'nonce': nonce, 'pid': host_pid})
                        store.update(key, script_host_pid=host_pid)
                    except (OSError, ValueError, KeyError):
                        pass
                if _read_event(channel, 'started', nonce) and not handoff:
                    handoff = True
                    store.update(key, handoff=True)
                if time.monotonic() >= deadline:
                    timed_out = True
                    store.update(key, term_at=time.time())
                    mailbox.write(channel, 'terminate.json', {'nonce': nonce})
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                    break
                time.sleep(.2)
            code = process.wait()
            # Retain the exclusive lease until the whole supervised scope exits.
            from .runtime import members
            while members(store.get(key)):
                time.sleep(.2)
        handoff = handoff or bool(_read_event(channel, 'started', nonce))
        result = _read_event(channel, 'finished', nonce)
        timed_out = timed_out or 'term_at' in store.get(key)
        abnormal = timed_out or code < 0 or code >= 128 or handoff and result is None
        success = handoff and result is not None and result.get('success') is True and code == 0 and not abnormal
        error = 'Execution timed out' if timed_out else f'Process exit={code}; script result={result.get("status") if result else "unavailable"}'
        kind = 'interrupted' if abnormal else 'startup_failed' if not handoff else None
        store.update(key, event_kind=kind, error_tail=_error_tail(log, spec['params']) if kind else '', phase='success' if success else 'unknown' if abnormal else 'failed',
                              finished_at=time.time(), returncode=code, handoff=handoff,
                              error='' if success else error, script_result=result,
                              script_mail=_read_event(channel, 'mail', nonce) or {'enabled': spec['params']['email'], 'delivered': None})
    except Exception as exc:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        if process is not None and store.get(key).get('pid'):
            from .runtime import members
            try:
                if members(store.get(key)):
                    store.event(key + ':interrupted', dict(store.get(key), kind='interrupted', error=f'{type(exc).__name__}: {exc}; surviving processes require reconciliation'))
                    return  # Keep the active record blocking admission.
            except (OSError, RuntimeError, psutil.Error):
                return  # Unknown process scope must fail closed.
        store.update(key, event_kind='startup_failed' if not handoff else 'interrupted', phase='failed' if process is None else 'unknown', finished_at=time.time(),
                              handoff=handoff, log_path=str(log), error=f'{type(exc).__name__}: {exc}')


def recorded_task_blocker():
    from src.api.util.backend.task import TaskDatabase
    database = TaskDatabase()
    with database.conn_handler as (_, cursor):
        cursor.execute("SELECT task_id FROM task_records WHERE status IN ('starting','running') LIMIT 1")
        row = cursor.fetchone()
        return f'Recorded script {row[0]} is still active or awaiting reconciliation' if row else None


def control_failure(error, sender):
    """Separate alert ledger survives a missing/corrupt execution database."""
    import sqlite3
    directory = state_dir().parent
    directory.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(f'{type(error).__name__}:{error}'.encode()).hexdigest()
    with sqlite3.connect(directory / 'autofix-control-alerts.sqlite') as db:
        db.execute('CREATE TABLE IF NOT EXISTS notices (id TEXT PRIMARY KEY, sent INTEGER NOT NULL DEFAULT 0)')
        db.execute('INSERT OR IGNORE INTO notices(id) VALUES (?)', (key,))
        if db.execute('SELECT sent FROM notices WHERE id=?', (key,)).fetchone()[0]:
            return
        sent = sender('Autofix control failure - ' + socket.gethostname(),
                      f'Autofix is paused. No execution ledger has been recreated.\n{type(error).__name__}: {error}\nRestore the existing control state before resuming.',
                      confirmation_message='Learndl autofix control alert', queue_on_failure=False)
        if sent:
            db.execute('UPDATE notices SET sent=1 WHERE id=?', (key,))


def worker_observation(store, status, reason, *, code=0, **fields):
    observation = {'checked_at': time.time(), 'status': status, 'reason': reason, **fields}
    if store is not None:
        with store.connect() as db:
            db.execute("INSERT OR REPLACE INTO metadata VALUES ('worker_observation', ?)", (canonical(observation),))
    print('Autofix worker: ' + canonical(observation), flush=True)
    return code


def worker():
    from .maintenance_gate import MaintenanceBusy, configured, permit
    from .git_update import training_blocker
    from src.proj.util.script.code_update_lock import training_code_guard
    if not configured():
        return worker_observation(None, 'disabled', 'Autofix activation is disabled; verify scheduling installation')
    from .runtime import process_cgroup
    if sys.platform != 'linux' or not (process_cgroup(os.getpid()) or '').endswith('/learndl-autofix.service'):
        raise RuntimeError('The autofix worker must run in its installed systemd service')
    store = Store(state_dir())
    try:
        with portalocker.Lock(str(store.directory / 'worker.lock'), mode='r+', timeout=0):
            if reconcile(store):
                return worker_observation(store, 'waiting', 'An earlier autofix still has an active or uncertain process scope')
            # Deferred calendar tasks are dispatched before maintenance work.
            if store.scheduled():
                return worker_observation(store, 'waiting', 'Deferred calendar tasks take priority; inspect learndl-deferred.service', queued=len(store.scheduled()))
            if not discover(store, ROOT):
                return worker_observation(store, 'configuration_error', 'Cannot discover tasks; inspect manifest error events', code=1)
            candidates = sorted((r for r in store.rows() if r['phase'] == 'queued'), key=lambda r: r['order'])
            if not candidates:
                (store.directory / 'pending').unlink(missing_ok=True)
                return worker_observation(store, 'idle', 'No eligible unattempted IDs; inspect task history for conflicts or terminal outcomes')
            (store.directory / 'pending').touch()
            with permit(exclusive=True, blocking=False), training_code_guard():
                blocker = training_blocker() or recorded_task_blocker()
                if blocker:
                    return worker_observation(store, 'waiting', blocker)
                # Current manifest is read again under the code-update guard.
                if not discover(store, ROOT):
                    return worker_observation(store, 'configuration_error', 'Manifest changed and is invalid', code=1)
                candidates = sorted((r for r in store.rows() if r['phase'] == 'queued'), key=lambda r: r['order'])
                if candidates:
                    worker_observation(store, 'dispatching', 'Exclusive maintenance permit acquired', task_id=candidates[0]['id'])
                    execute(store, ROOT, candidates[0])
                    outcome = store.get(candidates[0]['id'])
                    worker_observation(store, 'completed_attempt', outcome.get('error') or outcome['phase'], task_id=outcome['id'], phase=outcome['phase'])
            if not any(r['phase'] == 'queued' for r in store.rows()):
                (store.directory / 'pending').unlink(missing_ok=True)
    except (portalocker.LockException, MaintenanceBusy) as exc:
        return worker_observation(store, 'waiting', f'Worker/maintenance permit occupied: {type(exc).__name__}: {exc}')
    return 0


def _child():
    # Wait for the host PID before importing a script so Monitor never sees a
    # namespace-local PID and mistakes it for an unrelated host process.
    from .maintenance_gate import sandbox_request
    request = sandbox_request()
    if request is None:
        raise RuntimeError('Missing sandbox context')
    deadline = time.monotonic() + 30
    host = Path(request['channel']) / 'host.json'
    while not host.exists():
        if time.monotonic() > deadline:
            raise RuntimeError('Supervisor did not register sandbox host PID')
        time.sleep(.05)
    info = json.loads(host.read_text())
    if info['nonce'] != request['nonce']:
        raise RuntimeError('Invalid host registration')
    os.environ['LEARNDL_AUTOFIX_HOST_PID'] = str(info['pid'])
    sys.argv = sys.argv[2:]
    import runpy
    runpy.run_path(sys.argv[0], run_name='__main__')
    return 0


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '_child':
        return _child()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['validate', 'plan', 'status', 'worker'])
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    if args.action == 'worker':
        if args.root.resolve() != ROOT:
            parser.error('Worker root is fixed by installation')
        return worker()
    root = args.root.resolve()
    items = manifest(root / 'runs/autofix/tasks.yaml')
    directory = state_dir(root)
    store = Store(directory) if directory.exists() else None
    output, invalid = [], False
    for item in items:
        row = {'id': item['id'], 'enabled': item.get('enabled', True), 'execution': 'unknown (server ledger unavailable)'}
        try:
            row['request'] = redact(validate(root, item))
            row['valid'] = True
        except Exception as exc:
            row.update(valid=False, error=f'{type(exc).__name__}: {exc}')
            invalid = True
        if store:
            existing = store.get(item['id'])
            row['execution'] = existing['phase'] if existing else 'not registered on this host'
            row['record'] = redact(existing)
        output.append(row)
    result = {'schema_version': 1, 'state_directory': str(directory),
              'execution_state': 'known' if store else 'unknown (server ledger unavailable)', 'fixes': output}
    if args.action == 'status' and store:
        result['activation_enabled'] = store.enabled()
        with store.connect(readonly=True) as db:
            observation = db.execute("SELECT value FROM metadata WHERE key='worker_observation'").fetchone()
        result['last_worker_check'] = json.loads(observation[0]) if observation else None
        result['history'] = redact(store.rows())
        with store.connect(readonly=True) as db:
            result['scheduled'] = [dict(zip(('id', 'task', 'state'), row)) for row in db.execute('SELECT id,task,state FROM scheduled ORDER BY created')]
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(invalid)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as exc:
        print(f'Autofix: {type(exc).__name__}: {exc}', file=sys.stderr)
        raise SystemExit(1) from None
