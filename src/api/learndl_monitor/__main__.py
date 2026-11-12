"""Start or open the detached Learndl Monitor local web console."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen
from urllib.parse import urlsplit
import webbrowser

import portalocker

from .services import ROOT, STATE, Manager, SERVICES

__all__ = ['main']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--serve', action='store_true')
    parser.add_argument('--port', type=int, default=8500)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    STATE.mkdir(parents=True, exist_ok=True, mode=0o700)
    record = STATE / 'console.json'
    if not args.serve:
        with portalocker.Lock(str(STATE / 'launch.lock'), timeout=5):
            existing = None
            try:
                existing = json.loads(record.read_text())
                with urlopen(Request(existing['url'] + '/api/services', headers={'Authorization': 'Bearer ' + existing['token']}), timeout=2) as response:
                    assert json.load(response)['application'] == 'learndl-monitor'
            except (OSError, ValueError, AssertionError, KeyError):
                existing = None
            if not existing:
                with (STATE / 'console.log').open('ab') as log:
                    process = subprocess.Popen([sys.executable, '-m', 'src.api.learndl_monitor', '--serve', '--port', str(args.port)],
                                               cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=os.name != 'nt')
                import time
                for _ in range(50):
                    if process.poll() is not None:
                        raise RuntimeError('Console failed to start. See ' + str(STATE / 'console.log'))
                    try:
                        existing = json.loads(record.read_text())
                        if existing['pid'] == process.pid:
                            break
                    except (OSError, ValueError):
                        pass
                    time.sleep(.1)
                else:
                    raise RuntimeError('Console startup timed out')
        assert existing is not None
        url = existing['url'] + '/'
        console_port = urlsplit(existing['url']).port
        ports = sorted({console_port, *(service['port'] for service in SERVICES.values())})
        forwarding = ' '.join(f'-L {port}:127.0.0.1:{port}' for port in ports)
        sys.stdout.write(url + '\nSSH: ssh ' + forwarding + ' <host>\n')
        if not args.no_browser:
            webbrowser.open_new_tab(url)
        return
    token = secrets.token_urlsafe(32)
    manager = Manager()
    origin = f'http://127.0.0.1:{args.port}'

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def authorized(self):
            return (self.headers.get('Host') == f'127.0.0.1:{args.port}'
                    and self.headers.get('Origin', origin) == origin
                    and self.headers.get('Authorization') == 'Bearer ' + token)

        def reply(self, value, code=200):
            payload = json.dumps(value).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if urlsplit(self.path).path == '/':
                if (self.headers.get('Host') != f'127.0.0.1:{args.port}'
                        or self.headers.get('Origin', origin) != origin):
                    self.send_error(403)
                    return
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Referrer-Policy', 'no-referrer')
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Content-Security-Policy', "frame-ancestors 'none'; base-uri 'none'")
                self.end_headers()
                page = Path(__file__).with_name('index.html').read_text()
                self.wfile.write(page.replace('__CONSOLE_TOKEN__', token).encode())
                return
            if not self.authorized():
                self.reply({'error': 'Session expired. Refresh the page to reconnect.'}, 403)
                return
            if self.path == '/api/services':
                self.reply({'application': 'learndl-monitor', 'services': [manager.snapshot(key) for key in SERVICES]})
            elif self.path.startswith('/api/log/') and self.path.split('/')[-1] in SERVICES:
                path = STATE / (self.path.split('/')[-1] + '.log')
                if path.exists():
                    with path.open('rb') as stream:
                        stream.seek(max(0, path.stat().st_size - 16000))
                        content = stream.read().decode(errors='replace')
                else:
                    content = ''
                self.reply({'log': content})
            else:
                self.send_error(404)

        def do_POST(self):
            if not self.authorized() or self.path != '/api/action':
                self.send_error(403)
                return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size < 4096:
                    raise ValueError('Invalid request size')
                data = json.loads(self.rfile.read(size))
                if not isinstance(data, dict):
                    raise ValueError('Expected an object')
                manager.submit(data['service'], data['action'], data.get('pid'), data.get('created'))
                self.reply({'accepted': True}, 202)
            except (ValueError, KeyError) as exc:
                self.reply({'error': str(exc)}, 400)

    with portalocker.Lock(str(STATE / 'server.lock'), timeout=0):
        with ThreadingHTTPServer(('127.0.0.1', args.port), Handler) as server:
            record.write_text(json.dumps({'url': origin, 'token': token, 'pid': os.getpid()}))
            record.chmod(0o600)
            server.serve_forever()


if __name__ == '__main__':
    main()
