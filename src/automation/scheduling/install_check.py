"""Reconcile installation requirements and explicitly authorized business retiming."""
from __future__ import annotations

import copy
from contextvars import ContextVar
import difflib
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time
import uuid

import portalocker

from src.automation.console import logger
from src.automation.paths import ROOT, config_directory
from src.automation import version_patch
from . import installer, install_flow, install_request
from .config import installed_config, load_config, runtime_dir, schedule_slots
from .install_impact import sources
from . import retiming_privileged as privileged


RETIMING = ContextVar("authorized_retiming", default=False)


def permission_directory():
    return privileged.BASE / str(os.getuid())


def permission():
    path = permission_directory() / 'policy.json'
    if not path.exists():
        return None
    for item in (privileged.BASE, path.parent, path, path.parent / 'helper.py'):
        if item.is_symlink() or item.stat().st_uid != 0 or item.stat().st_mode & 0o022:
            raise ValueError('Unsafe retiming permission; authorize again')
    return json.loads(path.read_text())


def privileged_command(argv):
    request = {'argv': argv}
    if argv[:2] == ['sudo', 'install']:
        request['content'] = Path(argv[-2]).read_text()
    helper = permission_directory() / 'helper.py'
    result = subprocess.run(['sudo', '-n', '/usr/bin/python3', '-I', str(helper)],
                            input=json.dumps(request), text=True, capture_output=True, timeout=130)
    if result.returncode:
        raise RuntimeError(result.stderr or 'Authorized timer operation failed')
    return result.stdout


def context():
    path = runtime_dir() / 'installation-context.json'
    if path.exists():
        return json.loads(path.read_text())
    config = installed_config() or {}
    return {'host': config.get('host', socket.gethostname().split('.')[0]),
            'config_dir': str(config_directory(None, root=ROOT))}


def snapshot():
    directory = runtime_dir()
    location = context()
    cron = installer.read_cron()
    timezone = installer.command(['timedatectl', 'show', '--property=Timezone', '--value']).strip()
    plan = installer.build_plan(load_config(Path(location['config_dir']), location['host']),
                                cron, ROOT, Path(sys.executable), timezone)
    old = installed_config()
    if old and old.get('autofix_initialized'):
        plan['config']['autofix_initialized'] = True
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    actual = {name: (installer.UNIT_DIR / name).read_text() if (installer.UNIT_DIR / name).exists() else ''
              for name in set(plan['units']) | set(manifest.get('units', {}))}
    changes = {}
    def compare(name, before, after):
        if before != after:
            changes[name] = ''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                                       fromfile='installed/' + name, tofile='desired/' + name))
    compare('installed.json', json.dumps(old, indent=2, sort_keys=True) + '\n',
            json.dumps(plan['config'], indent=2, sort_keys=True) + '\n')
    for name in sorted(actual):
        compare(name, actual[name], plan['units'].get(name, ''))
        if name.endswith('.timer') and name in plan['units']:
            state = installer.timer_state(name)
            if state.get('ActiveState') != 'active' or state.get('UnitFileState') != 'enabled':
                changes[name + ' state'] = json.dumps(state, sort_keys=True) + ' -> active, enabled'
    compare('crontab', cron, plan['cron'])
    desktop = installer.desktop_path()
    compare('desktop-autostart', desktop.read_text() if desktop.exists() else '', plan['desktop_autostart'] or '')
    current_sources = sources()
    for name in sorted(manifest.get('sources', {}).keys() | current_sources.keys()):
        before, after = manifest.get('sources', {}).get(name), current_sources.get(name)
        if before != after:
            changes[name] = f'{before or "absent"} -> {after or "absent"}'
    # Copies outside the checkout must also match, even when manifest hashes do.
    from .autofix_store import state_dir
    for source, target in ((Path(__file__).with_name('worker_bootstrap.py'), state_dir() / 'worker_bootstrap.py'),
                           (Path(__file__).with_name('autofix_sandbox.py'), state_dir() / 'policy.py')):
        if not target.exists() or source.read_bytes() != target.read_bytes():
            changes[str(target)] = 'Installed copy differs from ' + str(source)
    conflicts = plan['conflicts'] or plan['global_conflicts']
    if conflicts:
        changes['configuration-conflicts'] = json.dumps(conflicts, sort_keys=True)
    return location, plan, old, manifest, cron, actual, changes


def retiming_tasks(old, new):
    """Every effective configuration difference must be a time on an existing schedule."""
    if not old or old == new:
        return set()
    before, after = copy.deepcopy(old), copy.deepcopy(new)
    old_schedules, new_schedules = before.pop('schedules'), after.pop('schedules')
    if before != after or old_schedules.keys() != new_schedules.keys():
        return set()
    changed = set()
    for key, schedule in new_schedules.items():
        previous = old_schedules[key]
        if schedule == previous:
            continue
        task = schedule['task']
        if task != previous['task'] or not old['tasks'][task]['enabled']:
            return set()
        allowed = {'days', 'times', 'every_minutes'}
        if {k: v for k, v in schedule.items() if k not in allowed} != {
                k: v for k, v in previous.items() if k not in allowed}:
            return set()
        if not schedule_slots(schedule) or not schedule_slots(previous):
            return set()
        changed.add(task)
    return changed


def authorized(snapshot_data, grant):
    location, plan, old, manifest, cron, actual, changes = snapshot_data
    tasks = retiming_tasks(old, plan['config'])
    if not grant or not tasks or not tasks.issubset(grant['tasks']):
        return False
    if any(old['tasks'][task] != grant.get('task_definitions', {}).get(task) for task in tasks):
        return False
    if grant['host'] != location['host'] or grant['config_dir'] != location['config_dir'] or grant['timezone'] != old['timezone']:
        return False
    if plan['impact']['mode'] != 'config' or plan['conflicts'] or plan['global_conflicts']:
        return False
    if set(plan['units']) != set(manifest['units']) or actual != manifest['units'] or cron != manifest['cron']:
        return False
    permitted = {'installed.json', 'crontab'} | {'learndl-schedule-' + task + '.timer' for task in tasks}
    if not set(changes).issubset(permitted):
        return False
    for name in set(changes) - {'installed.json', 'crontab'}:
        if name not in grant['timers'] or privileged.envelope(plan['units'][name]) != grant['timers'][name]['envelope']:
            return False
    return True


def clear_episode():
    from .git_update import _mail_store
    with _mail_store().connect() as db:
        db.execute('CREATE TABLE IF NOT EXISTS installation_check_state (id INTEGER PRIMARY KEY, digest TEXT)')
        db.execute('DELETE FROM installation_check_state')
    installer.atomic_json(runtime_dir() / 'installation-check.json', {'event': 'current'})


def notify(event, changes, location, *, applied=False):
    """One outbox entry per observed episode, irrespective of unrelated Git commits."""
    from .git_update import _mail_store
    directory = runtime_dir()
    path = directory / 'installation-check.json'
    digest = hashlib.sha256(json.dumps([event, changes], sort_keys=True).encode()).hexdigest()
    command = 'bash runs/install_schedules.sh '
    host = ' --host ' + shlex.quote(location['host']) + ' --config-dir ' + shlex.quote(location['config_dir'])
    head = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True, capture_output=True, timeout=10).stdout.strip()
    body = (f'Host: {location["host"]}\nCommit: {head}\nStatus: {event}\n'
            f'Timezone: {(installed_config() or {}).get("timezone", "unknown")}\n\n' +
            '\n\n'.join(f'{name}\n{diff}' for name, diff in sorted(changes.items())))
    if not applied:
        body += '\n\nPreview:\n' + command + 'plan' + host + ' --diff\nAuthorize installation:\n' + command + 'defer' + host
        if 'blocked' in event.lower() or 'failed' in event.lower():
            body += '\nInspect patch state:\n.venv/bin/python -m src.automation.version_patch status'
        body += '\nInspect installation state:\n' + command + 'pending' + host
        if 'transaction' in event or 'manual recovery' in event:
            body += '\nInspect before explicit rollback:\n' + command + 'status' + host
            body += '\nRestore the saved installation after inspection:\n' + command + 'rollback' + host
    key = 'installation-check-' + uuid.uuid4().hex
    data = {'kind': 'installation_check', 'subject': f'Watchdog installation - {event} - {location["host"]}', 'body': body}
    with _mail_store().connect() as db:
        db.execute('CREATE TABLE IF NOT EXISTS installation_check_state (id INTEGER PRIMARY KEY, digest TEXT)')
        db.execute('BEGIN IMMEDIATE')
        previous = db.execute('SELECT digest FROM installation_check_state WHERE id=1').fetchone()
        if previous and previous[0] == digest:
            return
        db.execute('INSERT OR REPLACE INTO installation_check_state VALUES (1, ?)', (digest,))
        db.execute('INSERT INTO git_update_mail(id,data) VALUES (?,?)', (key, json.dumps(data)))
    installer.atomic_json(path, {'digest': digest, 'event': event, 'notice_id': key, 'checked_at': time.time()})


def check():
    directory = runtime_dir()
    directory.mkdir(parents=True, exist_ok=True)
    with portalocker.Lock(str(directory / 'installation-check.lock'), timeout=0):
        location = context()
        try:
            with version_patch.code_lease(directory, shared=True):
                data = snapshot()
                location, plan, old, manifest, cron, actual, changes = data
                if not changes:
                    clear_episode()
                    return
                transaction = directory / 'transaction.json'
                pending = install_request.current()
                if ((transaction.exists() and json.loads(transaction.read_text()).get('phase') != 'complete')
                        or pending.get('phase') in install_request.ACTIVE):
                    notify('blocked by installation transaction', changes, location)
                    return
                if not authorized(data, permission()):
                    notify('authorization required', changes, location)
                    return
                if version_patch.needs_migration(ROOT, directory.parent):
                    notify('blocked by version patch', changes, location)
                    return
                fingerprint = hashlib.sha256(json.dumps(changes, sort_keys=True).encode()).hexdigest()
                failure_path = directory / 'automatic-retiming-failure.json'
                failure = json.loads(failure_path.read_text()) if failure_path.exists() else {}
                if failure.get('fingerprint') == fingerprint:
                    notify('automatic retiming failed; manual recovery required',
                           dict(changes, error=failure['error']), location)
                    return
                # The existing installer owns publication, catch-up and rollback.
                with install_flow.admission(plan=plan):
                    fresh = snapshot()
                    if fresh[-1] != changes or not authorized(fresh, permission()):
                        return
                    approval = {'plan': plan, 'baseline': {'cron': cron}, **location}
                    token = RETIMING.set(True)
                    try:
                        install_flow.transact(approval)
                    except Exception as exc:
                        error = f'{type(exc).__name__}: {exc}'
                        installer.atomic_json(failure_path, {'fingerprint': fingerprint, 'error': error})
                        notify('automatic retiming failed; manual recovery required', dict(changes, error=error), location)
                        return
                    finally:
                        RETIMING.reset(token)
                notify('retiming completed', changes, location, applied=True)
        except (version_patch.CodeBusy, portalocker.exceptions.LockException, install_flow.MaintenanceBusy):
            return
        except Exception as exc:
            logger.error('Installation check failed: %s', exc)
            notify('check or retiming failed', {'error': f'{type(exc).__name__}: {exc}'}, location)


def authorize(action, host, config_dir):
    if sys.platform != 'linux' or os.getuid() == 0:
        raise ValueError('Run as the ordinary service user on Linux')
    directory = runtime_dir()
    location = {'host': host, 'config_dir': str(config_directory(config_dir, root=ROOT))}
    payload = dict(location, action=action, uid=os.getuid(), tasks=[], task_definitions={}, timers={}, timezone='')
    with install_flow.admission(plan={'impact': {'mode': 'config', 'groups': []}}):
        if action == 'authorize':
            data = snapshot()
            _, plan, old, manifest, cron, actual, changes = data
            if changes or context() != location:
                raise ValueError('Complete an installation for this host/config directory before authorizing retiming')
            payload['timezone'] = old['timezone']
            for task, config in old['tasks'].items():
                if config['enabled'] and any(row['task'] == task for row in old['schedules'].values()):
                    payload['tasks'].append(task)
                    payload['task_definitions'][task] = config
                    name = 'learndl-schedule-' + task + '.timer'
                    if name in actual:
                        payload['timers'][name] = {'envelope': privileged.envelope(actual[name])}
        logger.info(json.dumps(payload, indent=2))
        if input('Authorize these retiming permissions? [y/N] ').strip().lower() not in {'y', 'yes'}:
            return 0
        source = Path(privileged.__file__).read_text()
        frozen = source.rsplit("if __name__ == '__main__':", 1)[0]
        frozen += '\nsetup(json.loads(sys.stdin.read()), ' + repr(source) + ')\n'
        subprocess.run(['sudo', '/usr/bin/python3', '-I', '-c', frozen], input=json.dumps(payload), text=True, check=True)
        installer.atomic_json(directory / 'installation-context.json', location)
    logger.info('Retiming permission %s complete', action)
    return 0
