"""Same-machine, read-only browsing of a storage report's SQLite snapshot."""
from __future__ import annotations

import hashlib
import json
import platform
import secrets
import sqlite3
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .storage_report import StorageNode, StorageScan, render_node


def machine_identity() -> str:
    """Stable host fingerprint; never include raw machine identifiers in the HTML."""
    machine_id = Path('/etc/machine-id')
    identity = machine_id.read_text().strip() if machine_id.is_file() else f'{platform.node()}:{uuid.getnode()}'
    return hashlib.sha256(identity.encode()).hexdigest()


class SnapshotIndex:
    """Stream directory rows into a single transaction; only finish marks it usable."""

    def __init__(self, path: Path):
        self.connection = sqlite3.connect(path)
        self.connection.executescript('''
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE nodes (
                node_id INTEGER PRIMARY KEY, parent_id INTEGER, path TEXT NOT NULL,
                size INTEGER NOT NULL, own INTEGER NOT NULL, complete INTEGER NOT NULL,
                target TEXT, is_dir INTEGER NOT NULL, has_children INTEGER NOT NULL
            );
            CREATE INDEX nodes_parent ON nodes(parent_id);
        ''')

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.connection.close()

    def add(self, node: StorageNode, parent_id: int | None) -> None:
        self.connection.execute('INSERT INTO nodes VALUES (?,?,?,?,?,?,?,?,?)', (
            node.node_id, parent_id, node.path, node.size, node.own, node.complete,
            node.target, node.is_dir, node.has_children,
        ))

    def finish(self, scan: StorageScan, content: str) -> None:
        metadata = {
            'version': '1', 'report_id': scan.report_id, 'machine': machine_identity(),
            'root': str(scan.root), 'started': scan.started, 'finished': scan.finished,
            'html_sha256': hashlib.sha256(content.encode('utf-8')).hexdigest(), 'ready': '1',
        }
        self.connection.executemany('INSERT INTO metadata VALUES (?,?)', metadata.items())
        self.connection.commit()


class StorageReportServer(HTTPServer):
    """One report per loopback session. The caller owns serve_forever/server_close."""

    def __init__(self, report: str | Path):
        report = Path(report).expanduser().resolve(strict=True)
        index = report.with_suffix('.sqlite3')
        if not index.is_file():
            raise ValueError('Missing snapshot index. Regenerate this report to enable dynamic browsing.')
        self.connection = sqlite3.connect(index.as_uri() + '?mode=ro', uri=True, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        try:
            self.connection.execute('PRAGMA query_only=ON')
            if self.connection.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
                raise ValueError('Corrupt snapshot index')
            metadata = dict(self.connection.execute('SELECT key, value FROM metadata'))
            content = report.read_bytes()
            if metadata.get('version') != '1' or metadata.get('ready') != '1':
                raise ValueError('Incomplete or unsupported snapshot index')
            if metadata.get('machine') != machine_identity():
                raise ValueError('Dynamic browsing is only available on the computer that generated this report')
            if metadata.get('html_sha256') != hashlib.sha256(content).hexdigest():
                raise ValueError('HTML report does not match its snapshot index')
            text = content.decode('utf-8')
            if f'<meta name="storage-report-id" content="{metadata["report_id"]}">' not in text:
                raise ValueError('Report identifier does not match its index')
            root = self.connection.execute('SELECT * FROM nodes WHERE parent_id IS NULL').fetchall()
            if len(root) != 1 or root[0]['path'] != '.' or not root[0]['is_dir']:
                raise ValueError('Invalid snapshot root')
            self.token = secrets.token_urlsafe(32)
            self.base = f'/{self.token}/'
            text = text.replace('const storageBase = null; // STORAGE_SESSION',
                                f'const storageBase = {json.dumps(self.base)}; // STORAGE_SESSION', 1)
            self.page = text.encode('utf-8')
            super().__init__(('127.0.0.1', 0), _ReportHandler)
        except BaseException:
            self.connection.close()
            raise
        self.origin = f'http://127.0.0.1:{self.server_port}'
        self.url = self.origin + self.base

    def server_close(self):
        super().server_close()
        self.connection.close()

    def children(self, node_id: int) -> dict:
        parent = self.connection.execute('SELECT * FROM nodes WHERE node_id=?', (node_id,)).fetchone()
        if parent is None or not parent['is_dir']:
            raise KeyError(node_id)
        children = self.connection.execute(
            'SELECT * FROM nodes WHERE parent_id=? ORDER BY size DESC, path', (node_id,)
        ).fetchall()
        nodes = [StorageNode(**{key: row[key] for key in row.keys() if key != 'parent_id'}) for row in children]
        return {'node_id': node_id, 'html': ''.join(render_node(node, parent['size']) for node in nodes)}


class _ReportHandler(BaseHTTPRequestHandler):
    server: StorageReportServer

    def log_message(self, *args):
        pass

    def reply(self, status: int, body: bytes, content_type: str = 'text/plain; charset=utf-8'):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Content-Security-Policy', "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        host = f'127.0.0.1:{self.server.server_port}'
        if (self.headers.get('Host') != host
                or self.headers.get('Origin', self.server.origin) != self.server.origin
                or self.headers.get('Sec-Fetch-Site') == 'cross-site'):
            self.reply(403, b'Local same-origin access only')
            return
        parsed = urlsplit(self.path)
        if parsed.path == self.server.base and not parsed.query:
            self.reply(200, self.server.page, 'text/html; charset=utf-8')
        elif parsed.path == self.server.base + 'children':
            try:
                query = parse_qs(parsed.query, strict_parsing=True)
                values = query.get('id', [])
                if set(query) != {'id'} or len(values) != 1 or not values[0].isascii() or not values[0].isdigit():
                    raise ValueError('Invalid node ID')
                node_id = int(values[0])
                if not 0 < node_id < 2**63:
                    raise ValueError('Invalid node ID')
                result = self.server.children(node_id)
            except (ValueError, KeyError):
                self.reply(404, b'Unknown node')
            except sqlite3.Error:
                self.reply(500, b'Unable to read snapshot index')
            else:
                self.reply(200, json.dumps(result).encode('utf-8'), 'application/json; charset=utf-8')
        else:
            self.reply(404, b'Not found')
