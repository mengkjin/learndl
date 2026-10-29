"""Preview/confirm once; install now or register an immutable one-shot background request."""
from __future__ import annotations

import argparse
import copy
import difflib
import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import portalocker
import psutil

from . import installer as install
from . import install_request as requests
from .autofix_store import initialize, state_dir
from .config import ENTRYPOINTS, load_config, runtime_dir
from .maintenance_gate import MaintenanceBusy, permit
from .runtime import boot_id


def broker_command(argv: list[str]) -> str:
    request: dict = {'argv': argv}
    if argv[:2] == ['sudo', 'install']:
        request['content'] = Path(argv[-2]).read_text()
    with socket.socket(socket.AF_UNIX) as client:
        client.connect(os.environ['LEARNDL_INSTALL_BROKER'])
        client.settimeout(130)
        client.sendall((json.dumps(request) + '\n').encode())
        response = json.loads(client.makefile('rb').readline(2_000_000))
    if 'error' in response:
        raise RuntimeError(response['error'])
    return response['output']


def baseline(root: Path, config_dir: Path, plan: dict) -> dict:
    """Only installation inputs; runtime task history may continue to change."""
    files = set(config_dir.rglob('*.yaml')) | set(Path(__file__).parent.glob('*.py'))
    files.update(root / path for path in ('src/api/calls/app.py', 'src/api/calls/launcher.py',
                                         'runs/install_schedules.sh'))
    files.update(runtime_dir() / name for name in ('manifest.json', 'installed.json'))
    files.add(install.desktop_path())
    manifest_path = runtime_dir() / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {'units': {}}
    names = set(plan['units']) | set(manifest['units']) | {'learndl-watchdog.service', 'learndl-watchdog.timer'}
    files.update(install.UNIT_DIR / name for name in names)
    return {'files': {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
                      for path in sorted(files)}, 'cron': install.read_cron(),
            'timezone': install.command(['timedatectl', 'show', '--property=Timezone', '--value']).strip()}


def preview(plan: dict, cron: str) -> None:
    print(json.dumps({k: v for k, v in plan.items() if k not in {'config', 'cron', 'units', 'desktop_autostart'}}, indent=2))
    print(''.join(difflib.unified_diff(cron.splitlines(True), plan['cron'].splitlines(True),
                                     fromfile='current crontab', tofile='approved crontab')))
    manifest_path = runtime_dir() / 'manifest.json'
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {'units': {}}
    for name in sorted(set(plan['units']) | set(previous['units'])):
        path = install.UNIT_DIR / name
        current = path.read_text() if path.exists() else ''
        print(''.join(difflib.unified_diff(current.splitlines(True), plan['units'].get(name, '').splitlines(True),
                                         fromfile='current/' + name, tofile='approved/' + name)))
    desktop = install.desktop_path()
    print(''.join(difflib.unified_diff((desktop.read_text() if desktop.exists() else '').splitlines(True),
                                     (plan.get('desktop_autostart') or '').splitlines(True),
                                     fromfile=str(desktop), tofile='approved autostart')))
    try:
        check_business_tasks(plan)
    except MaintenanceBusy as exc:
        print(f'Currently busy: {exc}')


def check_business_tasks(plan: dict) -> None:
    from .autofix import recorded_task_blocker
    from .git_update import training_blocker
    reason = training_blocker() or recorded_task_blocker()
    if reason:
        raise MaintenanceBusy(reason)
    from src.proj import PATH
    entrypoints = {str(PATH.main / value) for value in ENTRYPOINTS.values()}
    wrappers = {str(PATH.runs / name) for name in ('daily_update.sh', 'weekly_update.sh', 'rcquant_sec_backfill.sh')}
    for process in psutil.process_iter(['pid', 'cmdline']):
        argv = process.info.get('cmdline') or []
        if process.pid != os.getpid() and (entrypoints | wrappers).intersection(argv):
            waiter = state_dir() / f'waiting-{process.pid}.json'
            try:
                parked = json.loads(waiter.read_text())
                if parked['pid'] == process.pid and abs(parked['created'] - process.create_time()) < .01:
                    continue
            except (OSError, ValueError, KeyError, psutil.Error):
                pass
            raise MaintenanceBusy(f'Legacy script PID {process.pid} is running')


@contextmanager
def admission(request_id: str | None = None):
    directory = runtime_dir()
    directory.mkdir(parents=True, exist_ok=True)
    control = state_dir()
    try:
        with portalocker.Lock(str(directory / 'install.lock'), timeout=0), \
             portalocker.Lock(str(control / 'worker.lock'), mode='r+', timeout=0), \
             permit(directory=control, exclusive=True, blocking=False, installation_id=request_id):
            yield
    except portalocker.LockException as exc:
        raise MaintenanceBusy('Installation or maintenance lock occupied') from exc


def rollback_plan(backup: dict) -> dict:
    return {'config': backup['config'], 'units': backup['manifest']['units'], 'cron': backup['cron'],
            'conflicts': {}, 'global_conflicts': [], 'report': {},
            'desktop_autostart': backup.get('desktop_autostart'),
            'timer_states': backup['manifest'].get('timer_states', {})}


def transact(approval: dict) -> None:
    """Called only under admission. Ordinary errors never become a queued retry."""
    directory = runtime_dir()
    transaction_path = directory / 'transaction.json'
    before = transaction_path.read_bytes() if transaction_path.exists() else None
    try:
        plan = copy.deepcopy(approval['plan'])
        install._apply_plan(plan, approval['baseline']['cron'], directory)
        if plan['conflicts']:
            raise RuntimeError(f'Installation conflicts: {plan["conflicts"]}')
    except Exception:
        after = transaction_path.read_bytes() if transaction_path.exists() else None
        if after != before:
            backup = json.loads((directory / 'rollback.json').read_text())
            try:
                install._apply_plan(rollback_plan(backup), install.read_cron(), directory)
            except Exception as rollback_error:
                raise RuntimeError(f'Installation rollback failed: {rollback_error}') from rollback_error
        raise


def verify(approval: dict) -> None:
    if approval['boot_id'] != boot_id() or time.time() >= approval['expires_at']:
        raise ValueError('Installation authorization expired or server restarted')
    if baseline(Path(approval['root']), Path(approval['config_dir']), approval['plan']) != approval['baseline']:
        raise ValueError('Installation inputs changed after preview; preview and authorize again')
    transaction = runtime_dir() / 'transaction.json'
    if transaction.exists() and json.loads(transaction.read_text()).get('phase') != 'complete':
        raise ValueError('An unfinished installation transaction requires explicit rollback first')


def register(approval: dict) -> str:
    identifier = approval['id']
    bundle = Path('/run/learndl-install') / identifier
    row = {key: approval[key] for key in ('id', 'unit', 'boot_id', 'created_at', 'expires_at')}
    row.update(phase='registering', bundle=str(bundle))
    with requests.lock():
        if requests.current().get('phase') in requests.ACTIVE:
            raise RuntimeError('Another installation request is already pending')
        requests.save(row)
    try:
        with tempfile.TemporaryDirectory(prefix='learndl-authorized-install-') as temp:
            payload = Path(temp) / 'approval.json'
            payload.write_text(json.dumps(approval))
            # Pass the previewed helper as immutable argv bytes; never ask root
            # to import/execute a mutable project file, even during registration.
            source = approval['helper_source']
            frozen = f'from __future__ import annotations\n__source__ = {source!r}\n' + source.replace('from __future__ import annotations\n', '')
            try:
                subprocess.run(['sudo', '-n', '/usr/bin/python3', '-I', '-c', frozen,
                                'register', str(payload)], check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as exc:
                # CalledProcessError includes the entire frozen broker in argv.
                # Report its actual stderr instead of flooding the terminal.
                detail = (exc.stderr or exc.stdout or 'No diagnostic output').strip()
                raise RuntimeError(f'Background registration failed (exit {exc.returncode}):\n{detail[-4000:]}') from None
        # The root supervisor acknowledges creation of the unprivileged worker.
        for _ in range(100):
            if (bundle / 'ready').exists():
                with requests.lock():
                    latest = requests.current()
                    if latest.get('phase') == 'registering':
                        latest['phase'] = 'waiting'
                        requests.save(latest)
                print(f'Background installation registered: {identifier}\nService: {approval["unit"]}\n'
                      'You may close this CLI window or disconnect SSH.', flush=True)
                return identifier
            if (bundle / 'result.json').exists():
                raise RuntimeError('Background installer exited during registration')
            time.sleep(.1)
        raise RuntimeError('Background installer did not acknowledge registration')
    except BaseException:
        with requests.lock():
            latest = requests.current()
            if latest.get('id') == identifier and latest.get('phase') == 'registering':
                requests.finish(latest, 'failed', 'Background registration failed; no installation authorized to start')
        raise


def run_interactive(*, deferred: bool, host: str, config_dir: Path | None = None) -> int:
    from src.proj import PATH
    if sys.platform != 'linux' or os.getuid() == 0:
        raise RuntimeError('Run installation as the ordinary service user on Linux')
    existing = requests.reconcile()
    if existing.get('phase') in requests.ACTIVE:
        print(json.dumps(existing, indent=2))
        return 1
    root = PATH.main
    config_dir = (config_dir or root / 'runs/scheduling').resolve()
    cron = install.read_cron()
    timezone = install.command(['timedatectl', 'show', '--property=Timezone', '--value']).strip()
    plan = install.build_plan(load_config(config_dir, host), cron, root, Path(sys.executable), timezone)
    snapshot = baseline(root, config_dir, plan)
    helper_path = Path(__file__).with_name('install_privileged.py').resolve()
    helper_source = helper_path.read_bytes().decode('utf-8')
    if hashlib.sha256(helper_source.encode()).hexdigest() != snapshot['files'].get(str(helper_path)):
        raise ValueError('Privilege helper changed during planning; preview again')
    if snapshot['cron'] != cron or snapshot['timezone'] != timezone:
        raise ValueError('Server configuration changed during planning; preview again')
    preview(plan, cron)
    if plan['conflicts'] or plan['global_conflicts']:
        raise ValueError('Resolve preview conflicts before installation')
    for name, content in plan['units'].items():
        install.validate_unit(name, content)
    notice = ('If busy, authorize background installation, pause new managed tasks, and wait up to 24 hours. '
              'SSH disconnect is safe; reboot cancels waiting. ' if deferred else 'If busy, exit without installing. ')
    if input(notice + 'Apply these changes? [y/N] ').strip().lower() not in {'y', 'yes'}:
        print('Installation cancelled.')
        return 0
    subprocess.run(['sudo', '-v'], check=True)  # Terminal authentication before any admission change.
    control = state_dir()
    control.parent.mkdir(parents=True, exist_ok=True)
    with portalocker.Lock(str(control.parent / 'initialize.lock'), timeout=10):
        initialize(control, root)
    identifier = uuid.uuid4().hex
    names = set(plan['units']) | {Path(path).name for path in snapshot['files'] if Path(path).parent == install.UNIT_DIR}
    allowed = {}
    for name in names:
        path = install.UNIT_DIR / name
        old = path.read_bytes() if path.exists() else None
        actual = hashlib.sha256(old).hexdigest() if old is not None else None
        if actual != snapshot['files'].get(str(path.resolve())):
            raise ValueError('Unit contents changed since preview; authorize again')
        allowed[name] = list(dict.fromkeys(value for value in
                            (plan['units'].get(name), old.decode('utf-8') if old is not None else None)
                            if value is not None))
    approval = dict(id=identifier, unit=f'learndl-install-{identifier}.service', uid=os.getuid(),
                    root=str(root), python=sys.executable, config_dir=str(config_dir), plan=plan,
                    baseline=snapshot, allowed_units=allowed, boot_id=boot_id(),
                    helper_source=helper_source,
                    created_at=time.time(), expires_at=time.time() + 86400)
    # Only failure to acquire admission / running business tasks permits queuing.
    try:
        with admission():
            verify(approval)
            check_business_tasks(plan)
            try:
                transact(approval)
            except Exception as exc:
                transaction = runtime_dir() / 'transaction.json'
                if transaction.exists() and json.loads(transaction.read_text()).get('phase') != 'complete':
                    with requests.lock():
                        requests.finish(dict(id=identifier, boot_id=approval['boot_id']), 'repair_required', str(exc))
                raise
        print('Installation completed.')
        return 0
    except MaintenanceBusy as exc:
        if not deferred:
            print(f'Cannot install now: {exc}')
            return 1
        print(f'Cannot install now: {exc}; registering authorized background installation.')
    register(approval)
    return 0


def worker(bundle: Path) -> int:
    approval = json.loads((bundle / 'approval.json').read_text())
    identifier = approval['id']
    while True:
        row = requests.reconcile()
        if row.get('id') != identifier or row.get('phase') not in {'registering', 'waiting'}:
            break
        if row['phase'] == 'registering':
            time.sleep(.1)
            continue
        try:
            with admission(identifier):
                verify(approval)
                check_business_tasks(approval['plan'])
                with requests.lock():
                    row = requests.current()
                    if row.get('id') != identifier or row['phase'] != 'waiting':
                        break
                    row['phase'] = 'installing'
                    requests.save(row)
                try:
                    transact(approval)
                except Exception as exc:
                    transaction = runtime_dir() / 'transaction.json'
                    incomplete = transaction.exists() and json.loads(transaction.read_text()).get('phase') != 'complete'
                    with requests.lock():
                        requests.finish(row, 'repair_required' if incomplete else 'failed', str(exc))
                else:
                    with requests.lock():
                        requests.finish(row, 'success', 'Approved installation completed')
                break
        except MaintenanceBusy:
            time.sleep(1)
        except Exception as exc:
            with requests.lock():
                row = requests.current()
                if row.get('id') == identifier and row['phase'] == 'waiting':
                    requests.finish(row, 'cancelled', str(exc))
            break
    from src.proj.util.web.emailer import Email
    requests.deliver(Email.send)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['apply', 'defer', 'pending', 'cancel', 'worker'])
    parser.add_argument('bundle', nargs='?', type=Path)
    parser.add_argument('--host', default=socket.gethostname().split('.')[0])
    parser.add_argument('--config-dir', type=Path)
    args = parser.parse_args()
    if args.action in {'pending', 'cancel'}:
        print(json.dumps(requests.cancel() if args.action == 'cancel' else requests.reconcile(), indent=2))
        return 0
    if args.action == 'worker':
        if args.bundle is None or not os.environ.get('LEARNDL_INSTALL_BROKER'):
            parser.error('Worker requires a registered privilege broker')
        return worker(args.bundle)
    return run_interactive(deferred=args.action == 'defer', host=args.host, config_dir=args.config_dir)


if __name__ == '__main__':
    raise SystemExit(main())
