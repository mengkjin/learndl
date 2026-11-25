"""Loopback HTTP UI shared by the two artifact catalogs."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import json
import os
from pathlib import Path
import secrets
import shutil
from urllib.parse import parse_qs, quote, urlsplit
from zipfile import BadZipFile

from .artifact_library import LOCAL_DATA_TYPES, Library, open_folder, open_local
from .access import ADMIN_REQUIRED, is_guest, permits_guest_action, role_page
from .mail_html import static_mail_html
from .page_assets import PAGE_ASSETS, serve_page_asset
from .services import SERVICES, STATE


__all__ = ['main']


def main(kind, *, state=STATE):
    port = SERVICES[kind]['port']
    origin = f'http://127.0.0.1:{port}'
    token = secrets.token_urlsafe(32)
    catalog = Library(kind, state=state)
    title = SERVICES[kind]['zh']
    other_kind = 'model_preview' if kind == 'mail_archive' else 'mail_archive'
    other = SERVICES[other_kind]['port']

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def local(self):
            return self.headers.get('Host') == f'127.0.0.1:{port}' and self.headers.get('Origin', origin) == origin

        def authorized(self):
            return self.local() and hmac.compare_digest(self.headers.get('Authorization', ''), 'Bearer ' + token)

        def headers_for(self, status, mime, length, *, referrer='no-referrer'):
            self.send_response(status)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(length))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Referrer-Policy', referrer)

        def reply(self, value, status=200, html=False):
            output = (value if html else json.dumps(value, ensure_ascii=False)).encode()
            self.headers_for(status, 'text/html; charset=utf-8' if html else 'application/json', len(output),
                             referrer='same-origin' if html else 'no-referrer')
            self.send_header('Content-Security-Policy',
                             "default-src 'none'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
                             "connect-src 'self'; img-src data: blob:; frame-src blob:; base-uri 'none'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(output)

        def do_GET(self):
            url = urlsplit(self.path)
            if not self.local():
                return self.reply({'error': 'Forbidden'}, 403)
            if kind == 'model_preview' and url.path in ('/manage', '/api/model-management') and is_guest(self.headers):
                return self.reply({'error': ADMIN_REQUIRED}, 403)
            if url.path in PAGE_ASSETS:
                return serve_page_asset(self, url.path)
            if url.path == '/health':
                return self.reply({'ok': True})
            if url.path == '/' and kind == 'model_preview':
                self.send_response(302)
                self.send_header('Location', '/preview' + ('?' + url.query if url.query else ''))
                self.end_headers()
                return
            if url.path == '/' or kind == 'model_preview' and url.path in ('/preview', '/manage'):
                page = Path(__file__).with_name('artifact_library.html').read_text()
                return self.reply(role_page(page, self.headers).replace('__TOKEN__', token).replace('__TITLE__', title)
                                  .replace('__KIND__', kind).replace('__OTHER__', str(other)), html=True)
            if not self.authorized():
                return self.reply({'error': 'Forbidden'}, 403)
            params = parse_qs(url.query)
            def param(name, default=''):
                return params.get(name, [default])[0]

            try:
                if url.path == '/api/list':
                    return self.reply(catalog.listing(param('run')))
                if url.path == '/api/mail-headers' and kind == 'mail_archive':
                    return self.reply(catalog.mail_headers(catalog.available_row(param('id'))))
                if url.path == '/api/mail-html' and kind == 'mail_archive':
                    path, _, mime = catalog.file(catalog.available_row(param('id')), int(param('index')))
                    if mime != 'text/html':
                        raise ValueError('Select an HTML mail attachment')
                    limit = 16 * 1024 * 1024
                    with path.open('rb') as stream:
                        source = stream.read(limit + 1)
                    if len(source) > limit:
                        raise ValueError('HTML attachment exceeds the 16 MiB browser preview limit')
                    return self.reply({'html': static_mail_html(source.decode('utf-8-sig', errors='replace'))})
                if url.path == '/api/models' and kind == 'model_preview':
                    return self.reply(catalog.model_listing(param('run')))
                if url.path == '/api/model-management' and kind == 'model_preview':
                    from src.res.model.workflow.model_management import MOVE_JOURNAL, inventory

                    return self.reply({'records': list(inventory(catalog.management_context()).values()),
                                       'pending': (catalog.state / MOVE_JOURNAL).is_file()})
                if url.path == '/api/chapters':
                    row = catalog.available_row(param('id'))
                    return self.reply(catalog.chapters(row, max(0, int(param('page', '0'))), param('run')))
                if url.path == '/api/spreadsheet':
                    row = catalog.available_row(param('id'))
                    return self.reply(catalog.spreadsheet(row, int(param('index', '0')), param('sheet')))
                if url.path == '/api/file':
                    path, name, mime = catalog.file(catalog.available_row(param('id')), int(param('index', '0')))
                    with path.open('rb') as stream:
                        self.headers_for(200, mime, os.fstat(stream.fileno()).st_size)
                        self.send_header('Content-Disposition', "attachment; filename*=UTF-8''" + quote(name, safe=''))
                        self.send_header('Content-Security-Policy', "sandbox; default-src 'none'")
                        self.end_headers()
                        shutil.copyfileobj(stream, self.wfile, 1024 * 1024)
                    return
                self.reply({'error': 'Not found'}, 404)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except (OSError, KeyError, IndexError, ValueError, BadZipFile) as exc:
                self.reply({'error': str(exc)}, 409)

        def do_POST(self):
            if not self.authorized() or self.path != '/api/action':
                return self.reply({'error': 'Forbidden'}, 403)
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 64 * 1024:
                    raise ValueError('Invalid request size')
                data = json.loads(self.rfile.read(size))
                if not isinstance(data, dict):
                    raise ValueError('Expected an object')
                guest = is_guest(self.headers)
                if guest and not permits_guest_action(kind, self.path, data):
                    return self.reply({'error': ADMIN_REQUIRED}, 403)
                action = data['action']
                if action in ('history_models', 'undo_models') and kind == 'model_preview':
                    from src.res.model.workflow.model_management import operation_history, undo_latest

                    return self.reply(operation_history(catalog.management_context()) if action == 'history_models'
                                      else undo_latest(catalog.management_context(), apply=data.get('apply') is True, expected=data.get('operation_id')))
                if action in ('plan_models', 'apply_models') and kind == 'model_preview':
                    from src.res.model.workflow.model_management import manage

                    return self.reply(manage(catalog.management_context(), data['operation'], data['items'],
                                             apply=action == 'apply_models'))
                if action in ('archive', 'unarchive', 'delete', 'restore', 'purge') and kind == 'mail_archive':
                    ids = data['ids']
                    if not isinstance(ids, list) or len(ids) > 500 or not all(isinstance(i, str) for i in ids):
                        raise ValueError('Select up to 500 items')
                    return self.reply(catalog.manage(action, ids))
                if action == 'model_folder' and kind == 'model_preview':
                    path = catalog.model_folder(data['id'], data.get('build'))
                    return self.reply({'folder': open_folder(path) if data.get('open_locally') is True else str(path)})
                row = catalog.available_row(data['id'])
                if action == 'open_attachment' and kind == 'mail_archive':
                    return self.reply({'file': catalog.open_mail_attachment(row, int(data['index']))})
                if action == 'open_file' and kind == 'model_preview':
                    path, _, _ = catalog.file(row, 0)
                    if guest and path.suffix.lower() not in LOCAL_DATA_TYPES | {'.html', '.htm'}:
                        return self.reply({'error': 'This file type cannot be opened on the host by a guest.'}, 403)
                    return self.reply({'file': open_local(path)})
                if action == 'folder':
                    if row.get('trashed'):
                        raise ValueError('Restore the item to open it')
                    path = row['path']
                    return self.reply({'folder': open_folder(path) if data.get('open_locally') is True
                                       else str(path if path.is_dir() else path.parent)})
                return self.reply(catalog.experiment(row, action, guest=guest))
            except Exception as exc:
                self.reply({'error': str(exc)}, 409)

    with ThreadingHTTPServer(('127.0.0.1', port), Handler) as server:
        server.serve_forever()
