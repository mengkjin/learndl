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

import psutil

__all__ = ['ROOT', 'STATE', 'SERVICES', 'Manager']
ROOT = Path(__file__).resolve().parents[3]
STATE = ROOT / '.local_resources' / 'learndl_monitor'
SERVICES = {
    'tasks': {'zh': '任务中心', 'en': 'Task Center', 'port': 8502, 'entry': 'src/api/task_monitor/launch.py'},
    'storage': {'zh': '存储洞察', 'en': 'Storage Insights', 'port': 8503, 'entry': 'src.api.learndl_monitor.storage'},
    'research': {'zh': '研究工作台', 'en': 'Research Workbench', 'port': 8501, 'entry': 'src/api/interactive/launch.py'},
}
for _key, _definition in SERVICES.items():
    _definition['port'] = int(os.environ.get(f'LEARNDL_{_key.upper()}_PORT', str(_definition['port'])))


class Manager:
    """Serialize operations per service and rediscover real processes on every request."""

    def __init__(self):
        STATE.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.locks = {key: threading.Lock() for key in SERVICES}
        self.status = {key: '' for key in SERVICES}
        self.errors = {key: '' for key in SERVICES}
        self.children: dict[str, subprocess.Popen] = {}

    def discover(self, key):
        entry = SERVICES[key]['entry']
        matches = []
        for process in psutil.process_iter(['pid', 'create_time', 'cmdline', 'username']):
            try:
                args = process.info['cmdline'] or []
                if process.info['username'] != psutil.Process().username():
                    continue
                if key == 'storage':
                    valid = '-m' in args and entry in args and Path(process.cwd()).resolve() == ROOT
                else:
                    valid = any(Path(arg).name == 'streamlit' for arg in args) and 'run' in args and any(
                        (Path(arg) if Path(arg).is_absolute() else Path(process.cwd()) / arg).resolve() == ROOT / entry
                        for arg in args if arg.endswith('.py'))
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
                try:
                    registered = json.loads((STATE / f'{key}.json').read_text())
                    if registered['pid'] == process.pid and registered['created'] == result['created']:
                        result['source'] = 'managed'
                except (OSError, ValueError, KeyError):
                    pass
                try:
                    ports = sorted({c.laddr.port for c in process.net_connections(kind='inet') if c.status == psutil.CONN_LISTEN})
                except psutil.AccessDenied:
                    ports = []
                if not ports and key != 'storage':
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
                            with urlopen(f'http://127.0.0.1:{port}/_stcore/health', timeout=1):
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
                    if action in ('stop', 'restart'):
                        process = matches[0]
                        process_created = process.create_time()
                        targets = process.children(recursive=True) + [process]
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
                        self.start(key)
            except Exception as exc:
                self.errors[key] = str(exc)
            finally:
                self.status[key] = ''
                self.locks[key].release()
        threading.Thread(target=work, daemon=True).start()

    def start(self, key):
        definition = SERVICES[key]
        port = definition['port']
        import socket
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(('127.0.0.1', port))
        if key == 'storage':
            args = [sys.executable, '-m', definition['entry']]
        else:
            args = [sys.executable, '-m', 'streamlit', 'run', str(ROOT / definition['entry']),
                    '--server.address', '127.0.0.1', '--server.port', str(port), '--server.headless', 'true']
        log = STATE / f'{key}.log'
        if log.exists() and log.stat().st_size > 5_000_000:
            log.replace(log.with_suffix('.log.old'))
        with log.open('ab') as output:
            child = subprocess.Popen(args, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
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
                    with urlopen(address if key == 'storage' else address + '/_stcore/health', timeout=1) as response:
                        if response.status == 200:
                            return
                except OSError:
                    pass
            time.sleep(.5)
        raise RuntimeError('Service is still starting; check logs and refresh')
