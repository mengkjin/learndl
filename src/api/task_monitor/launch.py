"""Serve the native Task Center on loopback."""

from __future__ import annotations

import argparse
import hmac
import json
import mimetypes
import os
import secrets
import shutil
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast
from urllib.parse import parse_qsl, quote, urlsplit

import psutil

from src.api.learndl_monitor.page_assets import PAGE_ASSETS, serve_page_asset
from src.api.task_monitor.web_api import TaskCenter

__all__ = ['create_server', 'main']

ASSETS = Path(__file__).parent
MAX_HTML_PREVIEW_BYTES = 4 * 1024 * 1024
PAGES = {'/', '/remote', '/finished'}
STATIC = {'/dashboard.js': 'dashboard.js', '/dashboard.css': 'dashboard.css',
          '/live_output.js': 'live_output.js'}


def create_server(port=8502, *, center=None):
    center = center if center is not None else TaskCenter()
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        @property
        def authority(self):
            return f'127.0.0.1:{cast(ThreadingHTTPServer, self.server).server_port}'

        def local(self):
            origin = f'http://{self.authority}'
            return self.headers.get('Host') == self.authority and self.headers.get('Origin', origin) == origin

        def authorized(self):
            return self.local() and hmac.compare_digest(self.headers.get('Authorization', ''), f'Bearer {token}')

        def headers_for(self, status, mime, length):
            self.send_response(status)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(length))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Referrer-Policy', 'same-origin')
            self.send_header('Content-Security-Policy',
                             "default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
                             "connect-src 'self'; img-src data:; frame-src 'self'; base-uri 'none'; frame-ancestors 'none'")

        def reply(self, value, status=200, mime='application/json'):
            data = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False).encode()
            self.headers_for(status, mime, len(data))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if not self.local():
                return self.reply({'error': 'Forbidden'}, 403)
            url = urlsplit(self.path)
            if url.path in PAGE_ASSETS:
                return serve_page_asset(self, url.path)
            if url.path in PAGES:
                page = (ASSETS / 'dashboard.html').read_text(encoding='utf-8')
                page = page.replace('__TOKEN__', token)
                return self.reply(page.encode(), mime='text/html; charset=utf-8')
            if url.path in STATIC:
                asset = ASSETS / STATIC[url.path]
                mime = mimetypes.guess_type(asset)[0] or 'application/octet-stream'
                return self.reply(asset.read_bytes(), mime=f'{mime}; charset=utf-8')
            if url.path == '/health':
                return self.reply({'ok': True, 'service': 'tasks'})
            if not self.authorized():
                return self.reply({'error': 'Forbidden'}, 403)
            try:
                params = dict(parse_qsl(url.query, keep_blank_values=True, max_num_fields=20))
                self.get_api(url.path, params)
            except LookupError as exc:
                self.reply({'error': str(exc)}, 404)
            except (ValueError, TypeError, psutil.Error) as exc:
                self.reply({'error': str(exc)}, 400)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except OSError as exc:
                self.reply({'error': str(exc)}, 503)

        def get_api(self, route, params):
            if route in {'/api/running', '/api/finished'}:
                return self.reply(center.local_list(route.rsplit('/', 1)[-1], params))
            if route == '/api/resources':
                return self.reply(center.resource_snapshot())
            if route == '/api/task':
                return self.reply(center.local_detail(params['task']))
            if route == '/api/remote':
                return self.reply(center.remote_list())
            if route == '/api/remote/task':
                return self.reply(center.remote_detail(params))
            if route in {'/api/output', '/api/remote/output'}:
                style = params.get('style', 'colored')
                if style not in {'colored', 'text'}:
                    raise ValueError('Unknown output style')
                reader = center.remote_output if route == '/api/remote/output' else center.local_output
                return self.reply(reader(params, style))
            if route in {'/api/report', '/api/download'}:
                return self.report(params, download=route.endswith('download'))
            self.reply({'error': 'Unknown endpoint'}, 404)

        def report(self, params, *, download):
            task = center.local_task(params['task'])
            source = center.source(task, int(params['source']))
            if task.is_active or source.suffix.lower() not in {'.html', '.htm'} or not source.is_file():
                raise LookupError('HTML report unavailable')
            with source.open('rb') as stream:
                if not download:
                    data = stream.read(MAX_HTML_PREVIEW_BYTES + 1)
                    if len(data) > MAX_HTML_PREVIEW_BYTES:
                        return self.reply({'error': 'Report exceeds the 4 MiB preview limit'}, 413)
                    return self.reply({'html': data.decode('utf-8', errors='replace')})
                size = os.fstat(stream.fileno()).st_size
                self.headers_for(200, 'application/octet-stream', size)
                self.send_header('Content-Disposition', f"attachment; filename*=UTF-8''{quote(source.name, safe='')}")
                self.end_headers()
                shutil.copyfileobj(stream, self.wfile)

        def do_POST(self):
            if not self.authorized() or self.headers.get('Origin') != f'http://{self.authority}':
                return self.reply({'error': 'Forbidden'}, 403)
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 16384:
                    raise ValueError('Invalid request length')
                params = json.loads(self.rfile.read(length))
                if not isinstance(params, dict):
                    raise ValueError('Invalid request')
                if self.path == '/api/kill/prepare':
                    return self.reply(center.prepare_termination(params['task']))
                if self.path == '/api/kill/confirm':
                    return self.reply(center.confirm_termination(params['token']))
                if self.path == '/api/remote/reload':
                    return self.reply(center.reload_remote(params))
                self.reply({'error': 'Unknown endpoint'}, 404)
            except (KeyError, LookupError, ValueError, TypeError, psutil.Error) as exc:
                self.reply({'error': str(exc)}, 400)
            except (BrokenPipeError, ConnectionResetError):
                pass

    return ThreadingHTTPServer(('127.0.0.1', port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=int(os.environ.get('LEARNDL_TASKS_PORT', '8502')))
    parser.add_argument('--open', action='store_true', help='Open Task Center in the browser')
    args = parser.parse_args()
    with create_server(args.port) as server:
        if args.open:
            webbrowser.open(f'http://127.0.0.1:{server.server_port}/')
        server.serve_forever()


if __name__ == '__main__':
    main()
