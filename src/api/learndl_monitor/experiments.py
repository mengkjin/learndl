"""Loopback experiment dashboard manager with persistent multi-instance discovery."""
from __future__ import annotations

import json
from datetime import datetime, timezone
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
from src.api.learndl_monitor.restart_control import restart_link

__all__ = ['Instances', 'main', 'argument', 'KINDS']
KINDS = {'tensorboard': ('tensorboard.main', 6006), 'optuna': ('optuna_dashboard', 8080)}


def argument(args, name, default=''):
    for index, value in enumerate(args):
        if value == name and index + 1 < len(args):
            return args[index + 1]
        if value.startswith(name + '='):
            return value.split('=', 1)[1]
    return default


def synced_sources(results, *, timestamps=False):
    """List experiment ZIPs from committed batches and model build indexes."""
    from src.automation.artifact_exchange.core import (
        Config, archived_batches, experiment_members, real_tree, received_member,
    )
    result = {'tensorboard': [], 'optuna': []}
    seen = set()
    results = results.resolve()
    config = Config(results=results)
    if not results.exists():
        return result
    for folder, manifest in archived_batches(config):
        if manifest['schema_version'] != 2:
            continue
        try:
            members = experiment_members(manifest)
            stamp = manifest.get('created_at', '')
            date = datetime.strptime(stamp, '%Y%m%dT%H%M%S%fZ').strftime('%Y-%m-%d %H:%M:%S UTC')
            label = ' · '.join([manifest['host'], manifest['category'], manifest['experiment'], date, manifest['status']])
            for name, kind in members.items():
                path = real_tree(folder / received_member(manifest, name, members, folder=folder))
                if path.is_file() and str(path) not in seen:
                    seen.add(str(path))
                    result[kind].append({'path': str(path), 'label': label + ' · ' + name, '_stamp': stamp})
        except (ValueError, KeyError, TypeError, OSError):
            continue
    from src.automation.artifact_exchange.model_builds import archived_builds

    for folder, index in archived_builds(config):
        for row in index['files'].values():
            kind = row['kind']
            if kind not in result:
                continue
            try:
                path = real_tree(folder / row['path'])
                stamp = row['stamp']
                if not re.fullmatch(r'[0-9]{8}T[0-9]{12}Z', stamp):
                    stamp = '19700101T000000000000Z'  # Migrated protocol v1 batch ID.
                datetime.strptime(stamp, '%Y%m%dT%H%M%S%fZ')
                if path.is_file() and str(path) not in seen:
                    seen.add(str(path))
                    label = ' · '.join([index['host'], index['model'], folder.name, row['path']])
                    result[kind].append({'path': str(path), 'label': label, '_stamp': stamp})
            except (ValueError, OSError):
                continue
    for items in result.values():
        items.sort(key=lambda item: (item['_stamp'], item['path']), reverse=True)
        for item in items:
            stamp = item.pop('_stamp')
            if timestamps:
                item['_timestamp'] = datetime.strptime(stamp, '%Y%m%dT%H%M%S%fZ').replace(tzinfo=timezone.utc).timestamp()
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
                                 archive_source=record.get('archive_source', '') if owned else '',
                                 port=port, display_source=str(Path(display).relative_to(ROOT)) if display and Path(display).is_relative_to(ROOT) else display, url=url if healthy else '', status='running' if healthy else 'not ready',
                                 origin='managed' if owned else 'external', controllable=True,
                                 error=record.get('error', '') if owned else '',
                                 ssh=f'ssh -L {port}:127.0.0.1:{port} <host>'))
            except (psutil.Error, OSError, ValueError):
                continue
        return sorted(rows, key=lambda row: (row['created'], row['pid']))

    def act(self, data):
        from src.automation.patch_services import interface_operation

        with interface_operation(self.state.parent, key='experiments'):
            return self._operate(data)

    def _operate(self, data):
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

    def start_for_maintenance(self, kind, source, *, port):
        """Start a previous instance while its caller holds the interface lock."""
        with portalocker.Lock(str(self.state / 'operations.lock'), timeout=1):
            return self._act({'action': 'start', 'kind': kind, 'source': source}, preferred_port=port)

    def _act(self, data, attempt=0, *, preferred_port=None):
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
        ports = (preferred_port,) if preferred_port else range(KINDS[kind][1], KINDS[kind][1] + 94)
        for port in ports:
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
                    return self._act(data, attempt + 1, preferred_port=preferred_port)
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
        from src.api.pkgs.dashboard import archived_tensorboard_index, _tensorboard_snapshot_ready, model_tensorboard, saved_optuna_databases

        def option(path, label):
            return {'path': str(path), 'label': label}

        bases = {p.parent.parent for pattern in ('*/*/results/tensorboard', '*/*/snapshot/tensorboard', '*/results/tensorboard', '*/snapshot/tensorboard')
                 for p in PATH.model.glob(pattern) if not p.is_relative_to(PATH.model_archive)}
        trained = [p for base in bases if _tensorboard_snapshot_ready(p := model_tensorboard(base))]
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
                  for stamp, p in dated]
        archived = archived_tensorboard_index(PATH.model_archive)
        databases = sorted(saved_optuna_databases(PATH), key=lambda p: p.stat().st_mtime, reverse=True)
        def optuna_label(path):
            return str(path.relative_to(PATH.optuna) if path.is_relative_to(PATH.optuna)
                       else path.relative_to(PATH.model))

        from src.automation.artifact_exchange.core import Config, archived_batches, experiment_members, real_tree
        config = Config.load()
        synchronized = synced_sources(config.results, timestamps=True)

        def latest(kind, local):
            candidates = local + [(item['_timestamp'], option(item['path'], item['label']))
                                  for item in synchronized[kind]]
            selected, seen = [], set()
            for _, item in sorted(candidates, key=lambda entry: (-entry[0], entry[1]['path'])):
                canonical = str(Path(item['path']).resolve())
                if canonical not in seen:
                    selected.append(item)
                    seen.add(canonical)
                if len(selected) == 3:
                    break
            return selected

        recent = latest('tensorboard', [(stamp, item) for (stamp, _), item in zip(dated, recent)])
        recent_optuna = latest('optuna', [(p.stat().st_mtime, option(
            p, f'{datetime.fromtimestamp(p.stat().st_mtime).astimezone():%Y-%m-%d %H:%M:%S %Z} · {optuna_label(p)}'))
            for p in databases])
        def synced(kind):
            return {'zh': '同步研究实验', 'en': 'Synchronized research experiments', 'items': [option(item['path'], item['label']) for item in synchronized[kind]]}

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
            folder = config.results / 'custom/legacy/model_experiments' / kind
            packages = sorted((p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() == '.zip'),
                              key=lambda p: (-p.stat().st_mtime, str(p)))
            received = []
            for batch, manifest in archived_batches(config):
                if manifest['schema_version'] == 1 and manifest['kind'] == kind:
                    for name in experiment_members(manifest):
                        path = real_tree(batch / name)
                        if path.is_file():
                            received.append(option(path, legacy_label(path, batch)))
            return {'zh': '下载的实验压缩包', 'en': 'Downloaded experiment ZIPs',
                    'items': [option(p, legacy_label(p, folder)) for p in packages] + received}
        groups = {
            'tensorboard': [
                {'zh': '最近训练', 'en': 'Latest training', 'items': recent},
                {'zh': '已训练模型', 'en': 'Trained models', 'items': [option(p, str(p.parent.parent.relative_to(PATH.model))) for p in sorted(trained)]},
                {'zh': '归档模型', 'en': 'Archived models', 'items': [option(model_tensorboard(p), f'{module} / {name} / {p.name}')
                    for module, names in archived.items() for name, paths in names.items() for p in paths]},
                synced('tensorboard'),
                downloaded('tensorboard'),
            ],
            'optuna': [
                {'zh': '最近实验', 'en': 'Latest study', 'items': recent_optuna},
                {'zh': '已保存实验', 'en': 'Saved studies', 'items': [option(p, optuna_label(p)) for p in databases]},
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
                page = Path(__file__).with_name('experiments.html').read_text().replace('__TOKEN__', token)
                self.reply(page.replace('__RESTART_LINK__', restart_link('experiments', port)), html=True)
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
