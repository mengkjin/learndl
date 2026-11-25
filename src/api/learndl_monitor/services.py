"""Service definitions, identity-checked discovery, and background operations."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from urllib.request import urlopen
from urllib.parse import urlsplit

import psutil

__all__ = ['ROOT', 'STATE', 'SERVICES', 'Manager']
ROOT = Path(__file__).resolve().parents[3]
STATE = ROOT / '.local_resources' / 'learndl_monitor'
SERVICES = {
    'tasks': {'zh': '任务中心', 'en': 'Task Center', 'port': 8502, 'entry': 'src.api.task_monitor.launch'},
    'storage': {'zh': '存储洞察', 'en': 'Storage Insights', 'port': 8503, 'entry': 'src.api.learndl_monitor.storage'},
    'research': {'zh': '研究工作台', 'en': 'Research Workbench', 'port': 8501, 'entry': 'src/api/interactive/launch.py'},
    'compare': {'zh': '模型对比', 'en': 'Model Compare', 'port': 8504, 'entry': 'src/api/learndl_monitor/compare_app.py'},
    'mail_archive': {'zh': '邮件档案', 'en': 'Mail Archive', 'port': 8506, 'entry': 'src.api.learndl_monitor.mail_archive'},
    'model_preview': {'zh': '模型预览', 'en': 'Model Preview', 'port': 8507, 'entry': 'src.api.learndl_monitor.model_preview'},
    'experiments': {'zh': '实验面板', 'en': 'Experiment Dashboards', 'port': 8505, 'entry': 'src.api.learndl_monitor.experiments'},
}
MODULE_SERVICES = ('tasks', 'storage', 'experiments', 'mail_archive', 'model_preview')
HEALTH_SERVICES = ('tasks', 'experiments', 'mail_archive', 'model_preview')

for _key, _definition in SERVICES.items():
    _definition['port'] = int(os.environ.get(f'LEARNDL_{_key.upper()}_PORT', str(_definition['port'])))


class Manager:
    """Serialize operations per service and rediscover real processes on every request."""

    def __init__(self):
        STATE.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.locks = {key: threading.Lock() for key in SERVICES}
        self.status = {key: '' for key in SERVICES}
        self.errors = {key: '' for key in SERVICES}
        self._restart_lock = threading.Lock()
        self._restart_batch = None
        self.children: dict[str, subprocess.Popen] = {}

    def discover(self, key):
        entry = SERVICES[key]['entry']
        matches = []
        for process in psutil.process_iter(['pid', 'create_time', 'cmdline', 'username']):
            try:
                args = process.info['cmdline'] or []
                if process.info['username'] != psutil.Process().username():
                    continue
                if key in MODULE_SERVICES:
                    valid = '-m' in args and entry in args and Path(process.cwd()).resolve() == ROOT
                else:
                    valid = any(Path(arg).name == 'streamlit' for arg in args) and 'run' in args and any(
                        (Path(arg) if Path(arg).is_absolute() else Path(process.cwd()) / arg).resolve() == ROOT / entry
                        for arg in args if arg.endswith('.py'))
                if not valid and key == 'tasks':
                    # Discover a pre-upgrade Streamlit instance so console Restart can replace it.
                    valid = ('run' in args and any(Path(arg).name == 'streamlit' for arg in args)
                             and any((Path(process.cwd()) / arg).resolve() == ROOT / 'src/api/task_monitor/launch.py'
                                     for arg in args if arg.endswith('.py')))
                if valid and process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                    matches.append(process)
            except (psutil.Error, OSError, ValueError):
                continue
        return sorted(matches, key=lambda p: (p.info['create_time'], p.pid))

    def snapshot(self, key):
        processes = self.discover(key)
        process = processes[0] if processes else None
        result = dict(SERVICES[key], id=key, state=self.status[key] or ('running' if process else 'stopped'),
                      error=self.errors[key], pid=process.pid if process else None, created=process.info['create_time'] if process else None, count=len(processes), url='', source='', uptime=0)
        if process:
            result['uptime'] = int(time.time() - float(process.info['create_time']))
            result['source'] = 'managed' if key in self.children and self.children[key].pid == process.pid else 'external'
            try:
                registered_port = None
                try:
                    registered = json.loads((STATE / f'{key}.json').read_text())
                    if registered['pid'] == process.pid and registered['created'] == result['created']:
                        result['source'] = 'managed'
                        registered_port = registered.get('port')
                except (OSError, ValueError, KeyError):
                    pass
                try:
                    ports = sorted({c.laddr.port for c in process.net_connections(kind='inet') if c.status == psutil.CONN_LISTEN})
                except psutil.AccessDenied:
                    ports = []
                if not ports and key in HEALTH_SERVICES:
                    args = process.cmdline()
                    port = registered_port or SERVICES[key]['port']
                    for index, arg in enumerate(args):
                        option, separator, value = arg.partition('=')
                        if option in {'--port', '--server.port'}:
                            if separator:
                                port = int(value)
                            elif index + 1 < len(args):
                                port = int(args[index + 1])
                    ports = [port]
                if not ports and key not in MODULE_SERVICES:
                    args = process.cmdline()
                    port = SERVICES[key]['port']
                    for i, arg in enumerate(args):
                        if arg == '--server.port' and i + 1 < len(args):
                            port = int(args[i + 1])
                        elif arg.startswith('--server.port='):
                            port = int(arg.split('=', 1)[1])
                    with urlopen(f'http://127.0.0.1:{port}/_stcore/health', timeout=1):
                        ports = [port]
                if ports and key != 'storage':
                    for port in ports:
                        try:
                            path = '/health' if key in HEALTH_SERVICES else '/_stcore/health'
                            if key == 'tasks' and SERVICES[key]['entry'] not in process.cmdline():
                                path = '/_stcore/health'
                            with urlopen(f'http://127.0.0.1:{port}{path}', timeout=1):
                                result['port'] = port
                                result['url'] = f'http://127.0.0.1:{port}'
                                break
                        except OSError:
                            continue
                if key == 'storage':
                    record = json.loads((STATE / 'storage.json').read_text())
                    if record['pid'] == process.pid and record['created'] == process.create_time():
                        result['url'] = record['url']
                        with urlopen(record['url'] + '/status', timeout=1) as response:
                            storage_status = json.load(response)
                        result['snapshot'] = storage_status.get('snapshot', '')
                        if storage_status['busy']:
                            result['state'] = 'scanning'
                        if storage_status['error']:
                            result['error'] = storage_status['error']
                if not result['url'] and not self.status[key]:
                    result['state'] = 'starting'
            except (psutil.Error, OSError, ValueError, KeyError):
                result['url'] = ''
                if not self.status[key]:
                    result['state'] = 'unavailable'
        return result

    def restart_status(self):
        with self._restart_lock:
            if self._restart_batch is None:
                return None
            return {'active': self._restart_batch['active'],
                    'services': self._restart_batch['services'].copy()}

    def restart_running(self, targets):
        if not isinstance(targets, list) or not 1 <= len(targets) <= len(SERVICES):
            raise ValueError('Select one or more running services')
        confirmed = {}
        for target in targets:
            if not isinstance(target, dict):
                raise ValueError('Invalid service identity')
            key, pid, created = target.get('id'), target.get('pid'), target.get('created')
            if (not isinstance(key, str) or key not in SERVICES or key in confirmed
                    or type(pid) is not int or type(created) not in (int, float)):
                raise ValueError('Invalid service identity')
            confirmed[key] = (pid, created)
        with self._restart_lock:
            if self._restart_batch and self._restart_batch['active']:
                raise ValueError('A service restart batch is already running')
            batch = {'active': True, 'services': {key: {'state': 'pending'} for key in confirmed}}
            self._restart_batch = batch

        def record(key, state, error=''):
            with self._restart_lock:
                batch['services'][key] = {'state': state, 'error': error}

        def run():
            # Interface operations share a lock with version patches. Wait for
            # each service to come back before submitting the next restart.
            try:
                for key, (pid, created) in confirmed.items():
                    try:
                        current = self.snapshot(key)
                        if current['state'] != 'running' or (current['pid'], current['created']) != (pid, created):
                            record(key, 'skipped')
                            continue
                        record(key, 'restarting')
                        worker = self.submit(key, 'restart', pid, created)
                        worker.join()
                        current = self.snapshot(key)
                        if current['state'] == 'running' and (current['pid'], current['created']) != (pid, created):
                            record(key, 'restarted')
                        else:
                            record(key, 'failed', current['error'] or 'Service did not become ready')
                    except Exception as exc:
                        record(key, 'failed', str(exc))
            finally:
                with self._restart_lock:
                    batch['active'] = False

        threading.Thread(target=run, daemon=True).start()
        with self._restart_lock:
            return {'active': batch['active'], 'services': batch['services'].copy()}

    def submit(self, key, action, identity, created=None):
        if not isinstance(key, str) or not isinstance(action, str):
            raise ValueError('Invalid service or operation')
        if key not in SERVICES or action not in ('start', 'stop', 'restart', 'scan'):
            raise ValueError('Unknown service or operation')
        if action == 'scan' and key != 'storage':
            raise ValueError('Scanning is only available for storage')
        if not self.locks[key].acquire(blocking=False):
            raise ValueError('An operation is already running')
        self.status[key] = {'start': 'starting', 'stop': 'stopping', 'restart': 'restarting', 'scan': 'scanning'}[action]
        self.errors[key] = ''
        def work():
            try:
                from src.automation.patch_services import interface_operation

                with interface_operation(STATE, key=key):
                    matches = self.discover(key)
                    if action in ('stop', 'restart', 'scan'):
                        if not matches or matches[0].pid != identity or matches[0].create_time() != created:
                            raise ValueError('Service identity changed; refresh and try again')
                    if action == 'scan':
                        record = json.loads((STATE / 'storage.json').read_text())
                        from urllib.request import Request
                        with urlopen(Request(record['control'], data=b'{}', method='POST'), timeout=5):
                            pass
                    else:
                        restart_port = None
                        if action in ('stop', 'restart'):
                            process = matches[0]
                            process_created = process.create_time()
                            if action == 'restart':
                                current = self.snapshot(key)
                                if current['pid'] != identity or current['created'] != created:
                                    raise ValueError('Service identity changed; refresh and try again')
                                restart_port = urlsplit(current.get('url', '')).port
                            targets = [process] if key in HEALTH_SERVICES else process.children(recursive=True) + [process]
                            if psutil.Process(process.pid).create_time() != process_created:
                                raise ValueError('Process identity changed')
                            for target in targets:
                                try:
                                    target.terminate()
                                except psutil.NoSuchProcess:
                                    pass
                            _, alive = psutil.wait_procs(targets, timeout=5)
                            for target in alive:
                                target.kill()
                            _, alive = psutil.wait_procs(alive, timeout=3)
                            if alive:
                                raise RuntimeError('Service could not be stopped')
                        if action in ('start', 'restart'):
                            if self.discover(key):
                                raise ValueError('A service instance already exists')
                            self.start(key, port=restart_port)
            except Exception as exc:
                self.errors[key] = str(exc)
            finally:
                self.status[key] = ''
                self.locks[key].release()
        worker = threading.Thread(target=work, daemon=True)
        worker.start()
        return worker

    def start(self, key, *, port=None):
        definition = SERVICES[key]
        port = definition['port'] if port is None else port
        import socket
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(('127.0.0.1', port))
        if key in MODULE_SERVICES:
            args = [sys.executable, '-m', definition['entry']]
        else:
            args = [sys.executable, '-m', 'streamlit', 'run', str(ROOT / definition['entry']),
                    '--server.address', '127.0.0.1', '--server.port', str(port), '--server.headless', 'true']
        log = STATE / f'{key}.log'
        if log.exists() and log.stat().st_size > 5_000_000:
            log.replace(log.with_suffix('.log.old'))
        with log.open('ab') as output:
            child = subprocess.Popen(args, cwd=ROOT, env={**os.environ, f'LEARNDL_{key.upper()}_PORT': str(port)}, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                                     start_new_session=os.name != 'nt',
                                     creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0)
        self.children[key] = child
        if key != 'storage':
            record = STATE / f'{key}.json'
            record.write_text(json.dumps({'pid': child.pid, 'created': psutil.Process(child.pid).create_time(),
                                          'root': str(ROOT), 'service': key, 'port': port, 'command': args}))
            record.chmod(0o600)
        for _ in range(120):
            if child.poll() is not None:
                raise RuntimeError(f'Service exited ({child.returncode}); see logs')
            address = self.snapshot(key)['url']
            if address:
                try:
                    with urlopen(address if key == 'storage' else address + ('/health' if key in HEALTH_SERVICES else '/_stcore/health'), timeout=1) as response:
                        if response.status == 200:
                            return
                except OSError:
                    pass
            time.sleep(.5)
        raise RuntimeError('Service is still starting; check logs and refresh')
