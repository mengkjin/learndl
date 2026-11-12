"""Persistent storage snapshot service with asynchronous rescan and email delivery."""
from __future__ import annotations

import json
import os
import secrets
import sqlite3
import threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import psutil

from .services import STATE, SERVICES

__all__ = ['main']


def main():
    from src.proj import PATH
    from src.api.calls.computer import ProjectStorageReport
    from src.proj.util.filesys.storage_browser import StorageReportServer
    STATE.mkdir(parents=True, exist_ok=True, mode=0o700)
    port = SERVICES['storage']['port']
    origin = f'http://127.0.0.1:{port}'
    lock = threading.Lock()
    active = []
    view_lock = threading.Lock()
    token = secrets.token_urlsafe(32)
    status = {'busy': False, 'error': '', 'snapshot': '', 'progress': [], 'revision': 0}
    reports = []

    def progress(message):
        status['progress'] = [*status['progress'][-499:], str(message)]

    def scan(email):
        try:
            result = ProjectStorageReport(root=PATH.production, email=email, progress=progress).run()
            if result is None:
                raise RuntimeError('Report generation was cancelled')
            if email and result['email_sent'] is False:
                status['error'] = '报告已生成，但邮件发送失败 / Report generated; email delivery failed'
            server = StorageReportServer(result['report'])
            threading.Thread(target=server.serve_forever, daemon=True).start()
            with view_lock:
                previous = active[:]
                active[:] = [server]
                reports[:] = [result['report']]
                status['snapshot'] = str(result['report'].name)
                status['revision'] += 1
            for old in previous:
                old.shutdown()
                old.server_close()
        except Exception as exc:
            status['error'] = str(exc)
            progress(str(exc))
        finally:
            status['busy'] = False
            lock.release()

    def send_email():
        try:
            from src.proj.util.web.emailer import Email
            if not reports:
                raise ValueError('No completed snapshot is available')
            progress('Sending current report…')
            sent = Email.send(
                'Project Storage Report', f'Latest storage snapshot: {reports[0].name}',
                attachments=[reports[0]], project_attachments=False,
                confirmation_message='Project Storage Report',
            )
            if not sent:
                raise RuntimeError('邮件发送失败 / Email delivery failed')
            progress('邮件已发送 / Email sent')
        except Exception as exc:
            status['error'] = str(exc)
            progress(str(exc))
        finally:
            status['busy'] = False
            lock.release()

    def submit(action, email=True):
        if not lock.acquire(blocking=False):
            return False
        status.update(busy=True, error='', progress=[])
        target = send_email if action == 'email' else lambda: scan(email)
        threading.Thread(target=target, daemon=True).start()
        return True

    for report in sorted((PATH.production / 'results/storage_reports').glob('*.html'), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            server = StorageReportServer(report)
            root = server.connection.execute(
                "SELECT value FROM metadata WHERE key='root'").fetchone()[0]
            if root != str(PATH.production.resolve()):
                server.server_close()
                continue
            status['snapshot'] = report.name
            reports[:] = [report]
            status['revision'] = 1
            active.append(server)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            break
        except (OSError, ValueError, sqlite3.Error):
            continue
    if not active:
        submit('scan', False)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def allowed(self):
            return (self.headers.get('Host') == f'127.0.0.1:{port}'
                    and self.headers.get('Origin', origin) == origin
                    and self.headers.get('Sec-Fetch-Site') != 'cross-site')

        def end_headers(self):
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.send_header('X-Content-Type-Options', 'nosniff')
            super().end_headers()

        def do_GET(self):
            if not self.allowed():
                self.send_error(403)
                return
            if self.path == '/' + token + '/status':
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(status).encode())
                return
            prefix = '/' + token + '/snapshot/'
            if self.path.startswith(prefix) and active:
                from urllib.request import urlopen
                with view_lock:
                    snapshot = active[0]
                    with urlopen(snapshot.url + self.path[len(prefix):], timeout=10) as response:
                        content = response.read()
                        content_type = response.headers.get('Content-Type', 'application/json')
                        if response.headers.get_content_type() == 'text/html':
                            content = content.decode().replace(snapshot.base, prefix).encode()
                self.send_response(200)
                self.send_header('Content-Type', content_type)
                self.end_headers()
                self.wfile.write(content)
                return
            if self.path != '/' + token:
                self.send_error(404)
                return
            page = Path(__file__).with_name('storage.html').read_text().replace('__SERVICE_BASE__', '/' + token)
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(page.encode())

        def do_POST(self):
            action = self.path.removeprefix('/' + token + '/')
            if not self.allowed() or self.path not in ('/' + token + '/scan', '/' + token + '/email'):
                self.send_error(403)
                return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 <= size < 4096:
                    raise ValueError('Invalid request size')
                data = json.loads(self.rfile.read(size)) if size else {}
                if not isinstance(data, dict) or not isinstance(data.get('email', True), bool):
                    raise ValueError('Invalid email option')
            except (ValueError, TypeError):
                self.send_error(400)
                return
            if action == 'email' and not reports:
                self.send_error(409, 'No completed report')
                return
            if not submit(action, data.get('email', True)):
                self.send_error(409, 'An operation is already running')
                return
            self.send_response(202)
            self.end_headers()

    with ThreadingHTTPServer(('127.0.0.1', port), Handler) as service:
        address = f'{origin}/{token}'
        record = STATE / 'storage.json'
        record.write_text(json.dumps({'pid': os.getpid(), 'created': psutil.Process().create_time(), 'url': address, 'control': address + '/scan'}))
        record.chmod(0o600)
        service.serve_forever()


if __name__ == '__main__':
    main()
