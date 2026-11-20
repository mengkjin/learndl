"""Pause and restore registered project interfaces during version migrations."""
from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from functools import partial
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import NotRequired, TypedDict, cast
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import portalocker
import psutil

from src.automation.paths import ROOT
from src.shared.file_ops import atomic_json

__all__ = ['PatchServices', 'interfaces_paused', 'interface_operation']

DIAGNOSTIC = ('console', 'tasks')
JOURNAL = 'version-maintenance.json'


class ProcessIdentity(TypedDict):
    pid: int
    created: float


class ServiceIdentity(ProcessIdentity):
    port: int | None


class Dashboard(ProcessIdentity):
    kind: str
    archive_source: str
    source: str
    port: int


class RestoreResult(TypedDict, total=False):
    phase: str
    restored: list[str]
    errors: dict[str, str]
    cli_note: str


class Journal(TypedDict):
    phase: str
    services: dict[str, ServiceIdentity]
    experiments: list[Dashboard]
    restored: list[str]
    mappings: dict[str, str]
    cli: NotRequired[ProcessIdentity]
    cli_note: NotRequired[str]
    errors: NotRequired[dict[str, str]]


def read_record(path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}


def interfaces_paused(root=ROOT, key=None):
    return _paused(root / '.local_resources' / 'learndl_monitor', key)


def _paused(state, key):
    phase = read_record(state / JOURNAL).get('phase')
    return phase == 'migrating' or (phase == 'blocked' and key not in DIAGNOSTIC)


def _interface_lock(state):
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    return portalocker.Lock(state / 'interface-operations.lock', timeout=1)


@contextmanager
def interface_operation(state, *, key=None):
    """Serialize operator actions with the snapshot and stop transition."""
    lock = _interface_lock(state)
    try:
        lock.acquire()
    except portalocker.exceptions.LockException as exc:
        raise ValueError('An interface operation or version patch maintenance is in progress') from exc
    try:
        if _paused(state, key):
            raise ValueError('Version patch maintenance has paused this interface')
        yield
    finally:
        lock.release()


def live_process(record):
    if not record:
        return None
    try:
        process = psutil.Process(record['pid'])
        if process.create_time() == record['created'] and process.status() != psutil.STATUS_ZOMBIE:
            return process
    except psutil.NoSuchProcess:
        pass
    return None


def stop_process(record, *, descendants=True):
    process = live_process(record)
    if process is None:
        return
    if process.pid in {os.getpid(), *(parent.pid for parent in psutil.Process().parents())}:
        raise RuntimeError(f'Maintenance cannot stop its own terminal or parent process ({process.pid})')
    targets = process.children(recursive=True) if descendants else []
    targets = [*reversed(targets), process]
    for target in targets:
        try:
            target.terminate()  # psutil verifies process creation time before signalling.
        except psutil.NoSuchProcess:
            continue
    _, alive = psutil.wait_procs(targets, timeout=8)
    if alive:
        raise RuntimeError(f'Processes did not stop after SIGTERM: {[proc.pid for proc in alive]}')


class PatchServices:
    """Journal the original interfaces before stopping them, across attempts."""

    def __init__(self, root, runtime):
        self.root = root
        self.state = root / '.local_resources' / 'learndl_monitor'
        self.journal = self.state / JOURNAL
        self.cli_path = runtime / 'cli_recovery' / 'state.json'
        self.state_data: Journal = cast(Journal, read_record(self.journal))
        self.manager = None
        self.instances = None

    def report(self) -> RestoreResult:
        if not self.state_data:
            return {}
        return {'phase': self.state_data['phase'], 'restored': self.state_data['restored'],
                'errors': self.state_data.get('errors', {}), 'cli_note': self.state_data.get('cli_note', '')}

    def reconcile_needed(self):
        return self.state_data.get('phase') == 'migrating'

    def _monitor(self):
        from src.api.learndl_monitor.services import Manager
        from src.api.learndl_monitor.experiments import Instances

        if self.manager is None:
            self.manager = Manager()
        if self.instances is None:
            self.instances = Instances(self.state)
        return self.manager, self.instances

    def _console(self) -> ServiceIdentity | None:
        path = self.state / 'console.json'
        record = read_record(path)
        if not record:
            return None
        try:
            process = psutil.Process(record['pid'])
            started = process.create_time()
            args = process.cmdline()
            port = urlsplit(record['url']).port
            if ('src.api.learndl_monitor' not in args or '--serve' not in args or port is None
                    or Path(process.cwd()).resolve() != self.root.resolve()
                    or not process.is_running() or process.status() == psutil.STATUS_ZOMBIE):
                return None
            if record.get('created', started) != started or started > path.stat().st_mtime + 1:
                return None
            return {'pid': process.pid, 'created': started, 'port': port}
        except (psutil.NoSuchProcess, FileNotFoundError):
            return None

    def _snapshot(self) -> Journal:
        from src.api.learndl_monitor.services import SERVICES

        manager, instances = self._monitor()
        services: dict[str, ServiceIdentity] = {}
        for key in SERVICES:
            record = read_record(self.state / f'{key}.json')
            if not record:
                continue
            for process in manager.discover(key):
                if process.pid == record.get('pid') and process.create_time() == record.get('created'):
                    services[key] = {'pid': process.pid, 'created': record['created'],
                                     'port': record.get('port') or urlsplit(record.get('url', '')).port}
                    break
        if console := self._console():
            services['console'] = console
        experiments = [cast(Dashboard, row) for row in instances.discover() if row['origin'] == 'managed'] if any(
            instances.state.glob('[0-9]*.json')) else []
        return {'services': services, 'experiments': experiments, 'restored': [], 'mappings': {}, 'phase': 'migrating'}

    def pause(self):
        from src.automation import cli_recovery

        with _interface_lock(self.state):
            existing = cast(Journal, read_record(self.journal))
            if existing and existing.get('phase') == 'ready':
                raise RuntimeError('Finish interface recovery with version_patch restore-services before starting another migration')
            self.state_data = existing or self._snapshot()
            if sys.platform == 'linux' and self.cli_path.exists():
                with cli_recovery.lock(self.cli_path):
                    self._pause_cli(cli_recovery)
            else:
                self._save_migrating()
            for item in self.state_data['experiments']:
                stop_process(item)
            manager, _ = self._monitor()
            services = self.state_data['services']
            stop_order = [key for key in reversed(services) if key != 'console']
            if 'console' in services:
                stop_order.append('console')
            for key in stop_order:
                if key == 'console':
                    replacement = self._console()
                else:
                    current = read_record(self.state / f'{key}.json')
                    replacement = next((current for proc in manager.discover(key) if
                                        proc.pid == current.get('pid') and proc.create_time() == current.get('created')), {})
                stop_process(replacement or services[key], descendants=False)
                if key in self.state_data['restored']:
                    self.state_data['restored'].remove(key)
                    atomic_json(self.journal, self.state_data)

    def _save_migrating(self):
        self.state_data['phase'] = 'migrating'
        atomic_json(self.journal, self.state_data)

    def _pause_cli(self, recovery):
        state = recovery.load(self.cli_path)
        process = state.get('process', {})
        same_project = state.get('root') == str(self.root.resolve())
        if same_project and recovery.hub_alive(state) and not self.state_data.get('cli'):
            if not state.get('menu_idle') or not state.get('desired') or state.get('paused') or state.get('failure_paused'):
                # The existing CLI can stay open. Patches still check their files.
                self.state_data['cli_note'] = 'Registered CLI did not report an idle menu and was left running'
            elif live_process(process) is not None:
                protected = {os.getpid(), *(parent.pid for parent in psutil.Process().parents())}
                if process['pid'] not in protected:
                    self.state_data['cli'] = process
        self._save_migrating()
        if item := self.state_data.get('cli'):
            stop_process(item, descendants=False)

    def collect_paths(self, backup):
        """Record moves performed by the older experiment layout patch."""
        report = read_record(backup / 'experiment-results.json')
        for row in report.get('moves', []):
            if row.get('state') in ('migrated', 'already-migrated'):
                self.state_data['mappings'][row['source']] = row['target']
        if report:
            from src.automation.artifact_exchange.archive_paths import remap

            atomic_json(self.journal, self.state_data)
            mappings = {Path(src): Path(dst) for src, dst in self.state_data['mappings'].items()}
            for row in self.state_data['experiments']:
                path = self.state / 'experiments' / f'{row["pid"]}.json'
                record = read_record(path)
                if (record.get('created'), record.get('kind')) != (row['created'], row['kind']):
                    continue
                for key in ('archive_source', 'source'):
                    if record.get(key):
                        record[key] = str(remap(Path(record[key]), mappings))
                atomic_json(path, record)

    def _start_console(self, port):
        origin = f'http://127.0.0.1:{port}'
        with (self.state / 'console.log').open('ab') as log:
            child = subprocess.Popen([sys.executable, '-m', 'src.api.learndl_monitor', '--serve', '--port', str(port)],
                                     cwd=self.root, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                     start_new_session=True)
        for _ in range(50):
            if child.poll() is not None:
                break
            record = read_record(self.state / 'console.json')
            if record.get('pid') == child.pid:
                try:
                    self._console_health(record)
                    return
                except OSError:
                    pass
            time.sleep(.1)
        raise RuntimeError(f'Console startup failed at {origin}; see console.log')

    @staticmethod
    def _console_health(record):
        origin = record['url']
        request = Request(origin + '/api/services', headers={'Origin': origin,
                          'Authorization': 'Bearer ' + record['token']})
        with urlopen(request, timeout=3) as response:
            if json.load(response)['application'] != 'learndl-monitor':
                raise RuntimeError('Unexpected console service')

    def _restore_service(self, key: str, original: ServiceIdentity):
        manager, _ = self._monitor()
        if key == 'console':
            if self._console():
                self._console_health(read_record(self.state / 'console.json'))
            else:
                self._start_console(original['port'])
            return
        matches = manager.discover(key)
        if matches:
            record = read_record(self.state / f'{key}.json')
            if (not any(proc.pid == record.get('pid') and proc.create_time() == record.get('created') for proc in matches)
                    or not manager.snapshot(key)['url']):
                raise RuntimeError('Existing process is unregistered or not healthy')
        else:
            manager.start(key, port=original['port'])

    def _restore_experiment(self, row):
        from src.automation.artifact_exchange.archive_paths import remap

        _, instances = self._monitor()
        record = read_record(instances.state / f'{row["pid"]}.json')
        source = row.get('archive_source') or row['source']
        if record.get('created') == row['created'] and record.get('kind') == row['kind']:
            source = record.get('archive_source') or record.get('source') or source
        mapped = Path(source)
        if not mapped.exists():
            mappings = {Path(src): Path(dst) for src, dst in self.state_data['mappings'].items()}
            mapped = remap(mapped, mappings)
        mapped = mapped.resolve(strict=True)
        rows = [active for active in instances.discover() if active['origin'] == 'managed'
                and active['kind'] == row['kind'] and Path(active.get('archive_source') or active['source']) == mapped]
        if rows:
            if not any(active['status'] == 'running' for active in rows):
                raise RuntimeError('Existing dashboard is not healthy')
        else:
            result = instances.start_for_maintenance(row['kind'], str(mapped), port=row['port'])
            if not result.get('url'):
                raise RuntimeError('Dashboard startup did not report a healthy address')

    def restore(self, *, ready: bool) -> RestoreResult:
        if not self.journal.exists():
            return {}
        with _interface_lock(self.state):
            self.state_data = cast(Journal, read_record(self.journal))
            journal = self.state_data
            journal['phase'] = 'ready' if ready else 'blocked'
            errors = {}
            atomic_json(self.journal, journal)
            services = journal['services']
            actions: list[tuple[str, Callable[[], None]]] = [
                (key, partial(self._restore_service, key, services[key]))
                for key in dict.fromkeys((*DIAGNOSTIC, *services)) if key in services
                and (ready or key in DIAGNOSTIC)]
            if ready:
                actions.extend((f'{row["kind"]}:{row["pid"]}', partial(self._restore_experiment, row))
                               for row in journal['experiments'])
            for key, action in actions:
                if key in journal['restored']:
                    continue
                try:
                    action()
                except Exception as exc:
                    errors[key] = f'{type(exc).__name__}: {exc}'
                else:
                    journal['restored'].append(key)
                atomic_json(self.journal, journal)
            if ready and journal.get('cli'):
                from src.automation import cli_recovery

                try:
                    result = cli_recovery.tick(self.cli_path)
                    if result['status'] != 'alive':
                        errors['cli'] = result.get('error') or result['status']
                except (OSError, ValueError, RuntimeError, portalocker.exceptions.LockException) as exc:
                    errors['cli'] = str(exc)
            journal['errors'] = errors
            if ready and not errors:
                self.journal.unlink()
            else:
                atomic_json(self.journal, journal)
            return self.report()
