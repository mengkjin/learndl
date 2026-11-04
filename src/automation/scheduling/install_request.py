"""Durable installation admission and notifications, independent of autofix activation."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import portalocker

from .autofix_store import state_dir
from .runtime import boot_id

ACTIVE = {'registering', 'waiting', 'draining', 'installing', 'repair_required'}


def directory() -> Path:
    return state_dir().parent / 'installation'


def save(row: dict) -> None:
    from .installer import atomic_json
    target = directory()
    target.mkdir(parents=True, exist_ok=True)
    atomic_json(target / f'{row["id"]}.json', row)
    atomic_json(target / 'current.json', row)


def current() -> dict:
    path = directory() / 'current.json'
    return json.loads(path.read_text()) if path.exists() else {}


def lock():
    directory().mkdir(parents=True, exist_ok=True)
    return portalocker.Lock(str(directory() / 'request.lock'), timeout=1)


def finish(row: dict, phase: str, reason: str) -> dict:
    row.update(phase=phase, reason=reason, updated_at=time.time(), notified=False)
    if phase == 'repair_required':
        row['groups'] = None
    save(row)
    return row


def reconcile() -> dict:
    if not (directory() / 'current.json').exists():
        return {}
    with lock():
        row = current()
        if row.get('phase') not in ACTIVE or row['phase'] == 'repair_required':
            return row
        rebooted = row['boot_id'] != boot_id()
        result = Path(row['bundle']) / 'result.json' if row.get('bundle') else None
        died = bool(result and result.exists())
        if row.get('executor_pid'):
            import psutil
            try:
                executor = psutil.Process(row['executor_pid'])
                died = abs(executor.create_time() - row['executor_created']) >= .01 or executor.status() == psutil.STATUS_ZOMBIE
            except psutil.NoSuchProcess:
                died = True
            except psutil.Error:
                pass
        if not row.get('executor_pid') and not rebooted and not died and time.time() - row['created_at'] > 120:
            try:
                service = subprocess.run(['systemctl', 'show', row['unit'], '--property=ActiveState', '--value'],
                                         capture_output=True, text=True, timeout=5, check=False)
                died = service.returncode == 0 and service.stdout.strip() in {'inactive', 'failed'}
            except (OSError, subprocess.SubprocessError):
                pass  # Unreadable service state is not proof that its process stopped.
        if row['phase'] == 'installing' and (rebooted or died):
            return finish(row, 'repair_required', 'Installer interrupted during transaction; inspect rollback before reopening admission')
        if row['phase'] != 'installing':
            if rebooted:
                return finish(row, 'cancelled', 'Server restarted while installation was waiting')
            if time.time() >= row['expires_at']:
                return finish(row, 'cancelled', 'Installation authorization expired after 24 hours')
            if died:
                return finish(row, 'failed', 'Background installer exited before installation completed')
            if row['phase'] == 'registering' and time.time() - row['created_at'] > 120:
                return finish(row, 'failed', 'Background registration did not complete')
        return row


def pending() -> bool:
    return reconcile().get('phase') in ACTIVE


def cancel() -> dict:
    with lock():
        row = current()
        if row.get('phase') in {'draining', 'installing', 'repair_required'}:
            raise RuntimeError('Installation transaction must finish or be repaired before cancellation')
        if row.get('phase') in ACTIVE:
            finish(row, 'cancelled', 'Cancelled by user')
        return row


def deliver(sender) -> bool:
    reconcile()
    success = True
    if not directory().exists():
        return True
    for path in directory().glob('*.json'):
        if path.name == 'current.json':
            continue
        row = json.loads(path.read_text())
        waiting_notice = (row.get('phase') == 'waiting' and time.time() - row['created_at'] >= 3600
                          and time.time() - row.get('waiting_notified_at', 0) >= 86400)
        if not waiting_notice and (row.get('phase') in ACTIVE - {'repair_required'} or row.get('notified')):
            continue
        delivery = f'install:{row["id"]}:{row["phase"]}'
        if waiting_notice:
            delivery += ':' + str(int(time.time() // 86400))
        extra = ''
        files = []
        if waiting_notice:
            from .maintenance_notice import capture, render
            if not row.get('waiting_snapshot') or row.get('waiting_notified_at', 0) >= row['waiting_snapshot']['captured_at']:
                row['waiting_snapshot'] = capture(dict(row, owner='installation'))
                with lock():
                    latest = current()
                    if latest.get('id') == row['id'] and latest.get('phase') == 'waiting':
                        latest['waiting_snapshot'] = row['waiting_snapshot']
                        save(latest)
            extra = render(row['waiting_snapshot'])[1]
            from .maintenance_notice import attachment
            files = attachment(row['waiting_snapshot'])
        try:
            sent = sender(f'Watchdog installation - {row["phase"]}',
                          f'Request: {row["id"]}\nStatus: {row["phase"]}\n{row.get("reason", "")}\n'
                          f'Service: {row.get("unit", "")}\n' + extra,
                          confirmation_message='Learndl installation notification', queue_on_failure=False,
                          delivery_id=delivery, attachments=files)
        except Exception:
            sent = False
        if sent:
            with lock():
                latest = json.loads(path.read_text())
                if latest['phase'] == row['phase']:
                    if waiting_notice:
                        latest['waiting_notified_at'] = time.time()
                    else:
                        latest['notified'] = True
                    from .installer import atomic_json
                    atomic_json(path, latest)
                    if current().get('id') == row['id']:
                        save(latest)
        else:
            success = False
    return success
