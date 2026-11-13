"""Loopback experiment dashboard manager with persistent multi-instance discovery."""
from __future__ import annotations

import json
from datetime import datetime
import re
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import urlopen

import portalocker
import psutil

from src.api.learndl_monitor.services import ROOT, STATE, SERVICES

__all__ = ['Instances', 'main', 'argument', 'KINDS']
KINDS = {'tensorboard': ('tensorboard.main', 6006), 'optuna': ('optuna_dashboard', 8080)}


def argument(args, name, default=''):
    for index, value in enumerate(args):
        if value == name and index + 1 < len(args):
            return args[index + 1]
        if value.startswith(name + '='):
            return value.split('=', 1)[1]
    return default


def synced_sources(results):
    """Only committed v2 manifests can advertise synchronized experiment ZIPs."""
    from src.automation.artifact_exchange.core import (
        Config, _target, category_root, experiment_members, real_tree, safe_path, valid_category,
    )
    result = {'tensorboard': [], 'optuna': []}
    seen = set()
    results = results.resolve()
    config = Config(results=results)
    if not results.exists():
        return result
    for category in sorted(results.iterdir()):
        if not valid_category(category.name):
            continue
        try:
            root = category_root(results, category.name)
        except ValueError:
            continue
        for source in root.glob('*/*/source.json'):
            try:
                real_tree(source)
                manifest = json.loads(source.read_text())
                if manifest.get('schema_version') != 2 or manifest['category'] != category.name:
                    continue
                if _target(manifest, config) != source.parent:
                    continue
                members = experiment_members(manifest)
                stamp = manifest['created_at']
                date = datetime.strptime(stamp, '%Y%m%dT%H%M%S%fZ').strftime('%Y-%m-%d %H:%M:%S UTC')
                label = ' · '.join([manifest['host'], manifest['category'], manifest['experiment'], date, manifest['status']])
                for name, kind in members.items():
                    path = real_tree(source.parent / safe_path(name))
                    if path.is_file() and str(path) not in seen:
                        seen.add(str(path))
                        result[kind].append({'path': str(path), 'label': label + ' · ' + name, '_stamp': stamp})
            except (ValueError, KeyError, TypeError, OSError):
                continue
    for items in result.values():
        items.sort(key=lambda item: (item['_stamp'], item['path']), reverse=True)
        for item in items:
            item.pop('_stamp')
    return result


class Instances:
    """All mutations are serialized across requests and manager restarts."""

    def __init__(self, state=STATE):
        self.state = state / 'experiments'
        self.state.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.children = []

    def settings(self):
        path = self.state / 'limits.json'
        return json.loads(path.read_text()) if path.exists() else {'tensorboard': 5, 'optuna': 5}

    def discover(self):
        rows = []
        username = psutil.Process().username()
        for process in psutil.process_iter(['pid', 'create_time', 'cmdline', 'username']):
            try:
                args = process.info['cmdline'] or []
                if process.info['username'] != username or process.status() == psutil.STATUS_ZOMBIE:
                    continue
                # Match the actual executable or Python module, never a parent shell/uv command.
                names = [Path(a).name for a in args[:2]]
                module = argument(args, '-m')
                kind = 'tensorboard' if 'tensorboard' in names or module == 'tensorboard.main' else 'optuna' if 'optuna-dashboard' in names or module == 'optuna_dashboard' else None
                if kind is None:
                    continue
                source = argument(args, '--logdir') if kind == 'tensorboard' else next((a for a in args if a.startswith('sqlite:///')), '')
                if source.startswith('sqlite:///'):
                    source = source[len('sqlite:///'):]
                if source:
                    source = str((Path(process.cwd()) / source).resolve())
                port = int(argument(args, '--port', str(KINDS[kind][1])))
                try:
                    ports = [c.laddr.port for c in process.net_connections(kind='inet') if c.status == psutil.CONN_LISTEN]
                    if ports and port not in ports:
                        port = ports[0]
                except psutil.AccessDenied:
                    pass
                url = f'http://127.0.0.1:{port}'
                healthy = False
                try:
                    with urlopen(url, timeout=.5) as response:
                        healthy = response.status == 200
                except OSError:
                    pass
                record_path = self.state / f'{process.pid}.json'
                record = json.loads(record_path.read_text()) if record_path.exists() else {}
                owned = record.get('created') == process.info['create_time']
                display = record.get('archive_source') if owned else None
                display = display or source
                rows.append(dict(pid=process.pid, created=process.info['create_time'], kind=kind, source=source,
                                 port=port, display_source=str(Path(display).relative_to(ROOT)) if display and Path(display).is_relative_to(ROOT) else display, url=url if healthy else '', status='running' if healthy else 'not ready',
                                 origin='managed' if owned else 'external', controllable=True,
                                 error=record.get('error', '') if owned else '',
                                 ssh=f'ssh -L {port}:127.0.0.1:{port} <host>'))
            except (psutil.Error, OSError, ValueError):
                continue
        return sorted(rows, key=lambda row: (row['created'], row['pid']))

    def act(self, data):
        if data.get('action') == 'email':
            # Packaging/mail must not hold the process-control lock.
            from src.api.learndl_monitor.experiment_mail import send_experiment

            with portalocker.Lock(str(self.state / 'email.lock'), timeout=0):
                row = next((r for r in self.discover() if r['pid'] == data.get('pid')
                            and r['created'] == data.get('created') and r['kind'] == data.get('kind')), None)
                if row is None:
                    raise ValueError('Instance identity changed; refresh and retry')
                if not row['source']:
                    raise ValueError('Instance data source is unavailable')
                return {'mail_status': 'sent' if send_experiment(row['kind'], Path(row['source'])) else 'pending'}
        with portalocker.Lock(str(self.state / 'operations.lock'), timeout=1):
            return self._act(data)

    def _act(self, data, attempt=0):
        action = data.get('action')
        kind = data.get('kind')
        if kind not in KINDS:
            raise ValueError('Unknown dashboard type')
        if action == 'limit':
            value = data.get('value')
            if type(value) is not int or value < 1:
                raise ValueError('Limit must be a positive integer')
            limits = self.settings()
            limits[kind] = value
            temporary = self.state / 'limits.tmp'
            temporary.write_text(json.dumps(limits))
            temporary.replace(self.state / 'limits.json')
            return {'limits': limits}
        rows = self.discover()
        if action == 'stop_all':
            targets = data.get('targets')
            if not isinstance(targets, list):
                raise ValueError('Select the instances to stop')
            failures = []
            for target in targets:
                try:
                    self._act({'action': 'stop', 'kind': kind, 'pid': target['pid'], 'created': target['created']})
                except (ValueError, KeyError, TypeError, OSError, RuntimeError, psutil.Error) as exc:
                    failures.append(str(exc))
            if failures:
                raise RuntimeError('; '.join(failures))
            return {'stopped': len(targets)}
        if action == 'stop':
            row = next((r for r in rows if r['pid'] == data.get('pid') and r['created'] == data.get('created') and r['kind'] == kind), None)
            if row is None:
                raise ValueError('Instance identity changed; refresh and retry')
            process = psutil.Process(row['pid'])
            if process.create_time() != row['created']:
                raise ValueError('Process identity changed')
            targets = process.children(recursive=True) + [process]
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
                raise RuntimeError('Instance did not stop')
            return {'stopped': True}
        if action != 'start':
            raise ValueError('Unknown operation')
        source = Path(data['source']).expanduser().resolve(strict=True)
        archive_source = ''
        if source.suffix.lower() == '.zip':
            from src.api.learndl_monitor.experiment_archives import unpack_experiment

            archive_source = str(source)
            source = unpack_experiment(kind, source, self.state / 'imports')
        if (kind == 'tensorboard' and not source.is_dir()) or (kind == 'optuna' and not source.is_file()):
            raise ValueError('Select a log directory or SQLite file')
        if kind == 'optuna':
            with source.open('rb') as stream:
                if stream.read(16) != b'SQLite format 3\x00':
                    raise ValueError('Not a SQLite database')
        existing = next((r for r in rows if r['kind'] == kind and r['source'] == str(source)), None)
        if existing:
            return {'instance': existing, 'reused': True}
        if sum(row['kind'] == kind for row in rows) >= self.settings()[kind]:
            raise ValueError('Instance limit reached (including external instances)')
        for port in range(KINDS[kind][1], KINDS[kind][1] + 94):
            with socket.socket() as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    probe.bind(('127.0.0.1', port))
                    break
                except OSError:
                    continue
        else:
            raise RuntimeError('No free dashboard port')
        args = ([sys.executable, '-m', KINDS[kind][0]] if kind == 'tensorboard' else
                [sys.executable, str(Path(sys.executable).with_name('optuna-dashboard'))])
        args += ['--logdir', str(source)] if kind == 'tensorboard' else ['sqlite:///' + source.as_posix()]
        args += ['--host', '127.0.0.1', '--port', str(port)]
        log = self.state / f'{kind}-{time.time_ns()}.log'
        with log.open('ab') as output:
            child = subprocess.Popen(args, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=output, stderr=output,
                                     start_new_session=os.name != 'nt',
                                     creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0)
        self.children = [c for c in self.children if c.poll() is None] + [child]
        record = dict(pid=child.pid, created=psutil.Process(child.pid).create_time(), kind=kind, source=str(source), port=port, log=str(log))
        record['archive_source'] = archive_source
        path = self.state / f'{child.pid}.json'
        path.write_text(json.dumps(record))
        path.chmod(0o600)
        for _ in range(60):
            if child.poll() is not None:
                detail = log.read_text(errors='replace')[-3000:]
                if attempt < 2 and ('address already in use' in detail.lower() or 'port is in use' in detail.lower()):
                    return self._act(data, attempt + 1)
                raise RuntimeError(f'Dashboard exited: {detail}')
            try:
                with urlopen(f'http://127.0.0.1:{port}', timeout=.5) as response:
                    if response.status == 200:
                        return {'pid': child.pid, 'url': f'http://127.0.0.1:{port}'}
            except OSError:
                pass
            time.sleep(.5)
        record['error'] = 'Startup timed out; inspect logs or stop the instance'
        path.write_text(json.dumps(record))
        raise RuntimeError(record['error'])
    def sources(self):
        from src.proj import PATH
        from src.api.pkgs.dashboard import archived_tensorboard_index, _tensorboard_snapshot_ready

        def option(path, label):
            return {'path': str(path), 'label': label}

        trained = [p for p in PATH.model.glob('**/snapshot/tensorboard')
                   if not p.is_relative_to(PATH.model_archive) and _tensorboard_snapshot_ready(p)]
        def trained_time(path):
            stamps = []
            for event in path.rglob('*tfevents*'):
                try:
                    match = re.search(r'\.tfevents\.(\d+)(?:\.|$)', event.name)
                    stamp = int(match.group(1)) if match else event.stat().st_mtime
                    datetime.fromtimestamp(stamp)
                    stamps.append(stamp)
                except (OSError, ValueError, OverflowError):
                    continue
            return max(stamps) if stamps else path.stat().st_mtime

        dated = sorted(((trained_time(p), p) for p in trained), key=lambda item: (-item[0], str(item[1])))
        recent = [option(p, f'{datetime.fromtimestamp(stamp).astimezone():%Y-%m-%d %H:%M:%S %Z} · {p.parent.parent.relative_to(PATH.model)}')
                  for stamp, p in dated[:1]]
        archived = archived_tensorboard_index(PATH.model_archive)
        databases = sorted(PATH.optuna.rglob('*.sqlite3'), key=lambda p: p.stat().st_mtime, reverse=True)
        synchronized = synced_sources(ROOT / 'results')
        def synced(kind):
            return {'zh': '同步研究实验', 'en': 'Synchronized research experiments', 'items': synchronized[kind]}

        def legacy_label(path, folder):
            host = '来源未知'
            try:
                manifest = json.loads((path.parent / 'source.json').read_text())
                if isinstance(manifest.get('host'), str) and manifest['host']:
                    host = manifest['host']
            except (ValueError, OSError, AttributeError):
                pass
            return host + ' · ' + str(path.relative_to(folder))

        def downloaded(kind):
            folder = ROOT / 'results' / 'model_experiments' / kind
            packages = sorted((p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() == '.zip'),
                              key=lambda p: (-p.stat().st_mtime, str(p)))
            return {'zh': '下载的实验压缩包', 'en': 'Downloaded experiment ZIPs',
                    'items': [option(p, legacy_label(p, folder)) for p in packages]}
        groups = {
            'tensorboard': [
                {'zh': '最近训练', 'en': 'Latest training', 'items': recent},
                {'zh': '已训练模型', 'en': 'Trained models', 'items': [option(p, str(p.parent.parent.relative_to(PATH.model))) for p in sorted(trained)]},
                {'zh': '归档模型', 'en': 'Archived models', 'items': [option(p / 'snapshot/tensorboard', f'{module} / {name} / {p.name}')
                    for module, names in archived.items() for name, paths in names.items() for p in paths]},
                synced('tensorboard'),
                downloaded('tensorboard'),
            ],
            'optuna': [
                {'zh': '最近实验', 'en': 'Latest study', 'items': [option(p, str(p.relative_to(PATH.optuna))) for p in databases[:1]]},
                {'zh': '已保存实验', 'en': 'Saved studies', 'items': [option(p, str(p.relative_to(PATH.optuna))) for p in databases]},
                synced('optuna'),
                downloaded('optuna'),
            ],
        }

        for sections in groups.values():
            seen = set()
            for section in sections:
                if section['en'] not in ('Synchronized research experiments', 'Downloaded experiment ZIPs'):
                    continue
                items = []
                for item in section['items']:
                    canonical = str(Path(item['path']).resolve())
                    if canonical not in seen:
                        items.append(item)
                        seen.add(canonical)
                section['items'] = items
        return groups


def main():
    manager = Instances()
    token = secrets.token_urlsafe(32)
    port = SERVICES['experiments']['port']
    origin = f'http://127.0.0.1:{port}'

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def local(self):
            return self.headers.get('Host') == f'127.0.0.1:{port}' and self.headers.get('Origin', origin) == origin

        def reply(self, value, status=200, html=False):
            payload = value.encode() if html else json.dumps(value).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'text/html; charset=utf-8' if html else 'application/json')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Security-Policy', "frame-ancestors 'none'; base-uri 'none'")
            self.send_header('Referrer-Policy', 'no-referrer')
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if not self.local():
                self.reply({'error': 'Forbidden'}, 403)
            elif self.path == '/':
                self.reply(Path(__file__).with_name('experiments.html').read_text().replace('__TOKEN__', token), html=True)
            elif self.path == '/health':
                self.reply({'ok': True})
            elif self.headers.get('Authorization') != 'Bearer ' + token:
                self.reply({'error': 'Refresh the page to reconnect'}, 403)
            elif self.path == '/api/status':
                self.reply({'instances': manager.discover(), 'limits': manager.settings()})
            elif self.path == '/api/sources':
                self.reply(manager.sources())
            elif self.path.startswith('/api/log/'):
                try:
                    pid = int(self.path.rsplit('/', 1)[1])
                    record = json.loads((manager.state / f'{pid}.json').read_text())
                    with Path(record['log']).open('rb') as stream:
                        stream.seek(max(0, Path(record['log']).stat().st_size - 16000))
                        self.reply({'log': stream.read().decode(errors='replace')})
                except (ValueError, OSError, KeyError):
                    self.reply({'log': 'No managed log available'})
            else:
                self.reply({'error': 'Not found'}, 404)

        def do_POST(self):
            if not self.local() or self.path != '/api/action' or self.headers.get('Authorization') != 'Bearer ' + token:
                self.reply({'error': 'Forbidden'}, 403)
                return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size < 8192:
                    raise ValueError('Invalid request size')
                data = json.loads(self.rfile.read(size))
                if not isinstance(data, dict):
                    raise ValueError('Expected object')
                self.reply(manager.act(data))
            except Exception as exc:
                self.reply({'error': str(exc)}, 409)

    with portalocker.Lock(str(manager.state / 'server.lock'), timeout=0):
        with ThreadingHTTPServer(('127.0.0.1', port), Handler) as server:
            server.serve_forever()


if __name__ == '__main__':
    main()
