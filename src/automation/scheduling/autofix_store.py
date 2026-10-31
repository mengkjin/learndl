"""Protected, explicit-initialization ledger. Opening a missing DB never creates it."""
from __future__ import annotations

import hashlib
import json
import os
import socket
import sqlite3
import time
from pathlib import Path

from .autofix_spec import canonical, declaration, digest, redact

from src.automation.paths import ROOT
ACTIVE = {'starting', 'running'}


def state_dir(root: Path = ROOT) -> Path:
    key = hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:16]
    return Path.home() / '.local/state/learndl' / key / 'autofix'


def initialize(directory: Path, root: Path) -> None:
    """Installer only; never erase or repair an existing damaged ledger."""
    if directory.resolve().is_relative_to(root.resolve()):
        raise ValueError('Autofix control directory must be outside the writable project')
    if directory.exists():
        Store(directory).check()
        return
    directory.mkdir(parents=True, mode=0o700)
    (directory / 'logs').mkdir(mode=0o700)
    with sqlite3.connect(directory / 'ledger.sqlite') as db:
        db.executescript('''
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO metadata VALUES ('version', '1');
        INSERT INTO metadata VALUES ('enabled', 'false');
        CREATE TABLE fixes (id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, data TEXT NOT NULL);
        CREATE TABLE events (id TEXT PRIMARY KEY, data TEXT NOT NULL, sent INTEGER NOT NULL DEFAULT 0);
        CREATE TABLE scheduled (id TEXT PRIMARY KEY, task TEXT NOT NULL, state TEXT NOT NULL, created REAL NOT NULL);
        ''')
        db.execute('INSERT INTO metadata VALUES (?,?)', ('root', str(root.resolve())))
    for name in ('gate.lock', 'admission.lock', 'worker.lock'):
        (directory / name).touch(mode=0o600)
    (directory / 'policy.py').write_bytes(Path(__file__).with_name('autofix_sandbox.py').read_bytes())
    (directory / 'identity.json').write_text(canonical({'root': str(root.resolve()), 'host': socket.gethostname(), 'version': 1}))


class Store:
    def __init__(self, directory: Path):
        self.directory = directory
        self.path = directory / 'ledger.sqlite'
        self.check()

    def connect(self, *, readonly=False):
        db = sqlite3.connect(self.path.as_uri() + ('?mode=ro' if readonly else '?mode=rw'), uri=True, timeout=10)
        db.execute('PRAGMA busy_timeout=10000')
        return db

    def check(self):
        if self.directory != self.directory.resolve() or not (self.directory / 'logs').is_dir() or (self.directory / 'logs').is_symlink():
            raise RuntimeError('Autofix control directory/logs must be real directories outside the writable project')
        for name in ('ledger.sqlite', 'identity.json', 'gate.lock', 'admission.lock', 'worker.lock', 'policy.py'):
            path = self.directory / name
            if not path.is_file() or path.is_symlink():
                raise RuntimeError(f'Autofix state missing/unsafe: {path}; restore the ledger, do not recreate it')
        with self.connect(readonly=True) as db:
            if db.execute('PRAGMA quick_check').fetchone() != ('ok',):
                raise RuntimeError('Autofix ledger is corrupt; restore from backup')
            if db.execute("SELECT value FROM metadata WHERE key='version'").fetchone() != ('1',):
                raise RuntimeError('Unsupported autofix ledger version')
            root = Path(db.execute("SELECT value FROM metadata WHERE key='root'").fetchone()[0])
            identity = json.loads((self.directory / 'identity.json').read_text())
            if identity.get('root') != str(root) or self.directory.is_relative_to(root):
                raise RuntimeError('Autofix ledger identity/location mismatch')

    def enabled(self):
        with self.connect(readonly=True) as db:
            value = db.execute("SELECT value FROM metadata WHERE key='enabled'").fetchone()
        if value not in {('true',), ('false',)}:
            raise RuntimeError('Autofix activation state missing or corrupt; explicit installation required')
        return value == ('true',)

    def set_enabled(self, enabled):
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO metadata VALUES ('enabled', ?)", ('true' if enabled else 'false',))

    def rows(self):
        with self.connect(readonly=True) as db:
            return [json.loads(row[0]) for row in db.execute('SELECT data FROM fixes ORDER BY rowid')]

    def get(self, key):
        with self.connect(readonly=True) as db:
            row = db.execute('SELECT data FROM fixes WHERE id=?', (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def register(self, items, commit):
        conflicts = []
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            present = set()
            for order, item in enumerate(items):
                key = item['id']
                present.add(key)
                fingerprint = digest(declaration(item))
                previous = db.execute('SELECT fingerprint,data FROM fixes WHERE id=?', (key,)).fetchone()
                if previous:
                    record = json.loads(previous[1])
                    # Recompute from the frozen item so legacy email=False hashes
                    # adopt the mandatory-mail policy without resetting attempts.
                    record['conflict'] = digest(declaration(record['item'])) != fingerprint
                    if record['conflict']:
                        conflicts.append(key)
                    if not record.get('attempted'):
                        record['phase'] = 'queued' if item.get('enabled', True) is not False and not record['conflict'] else 'withdrawn'
                    record['order'] = order
                else:
                    record = {'id': key, 'item': item, 'phase': 'queued' if item.get('enabled', True) is not False else 'withdrawn',
                              'attempted': False, 'order': order, 'registered_at': time.time(), 'commit': commit, 'host': socket.gethostname()}
                db.execute('INSERT INTO fixes VALUES (?,?,?) ON CONFLICT(id) DO UPDATE SET data=excluded.data',
                           (key, fingerprint, canonical(record)))
            for key, raw in db.execute('SELECT id,data FROM fixes').fetchall():
                record = json.loads(raw)
                if key not in present and not record.get('attempted'):
                    record['phase'] = 'withdrawn'
                    db.execute('UPDATE fixes SET data=? WHERE id=?', (canonical(record), key))
        for key in conflicts:
            current = self.get(key)
            if not isinstance(current, dict):
                raise KeyError(key)
            self.event(f'{key}:conflict', dict(current, kind='configuration_error', error='ID declaration changed. Add a new ID; this ID will not be retried.'))

    def update(self, key, *, event_kind=None, **fields):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT data FROM fixes WHERE id=?', (key,)).fetchone()
            if not row:
                raise KeyError(key)
            record = json.loads(row[0])
            record.update(fields)
            db.execute('UPDATE fixes SET data=? WHERE id=?', (canonical(record), key))
            if event_kind:
                db.execute('INSERT OR IGNORE INTO events(id,data) VALUES (?,?)',
                           (key + ':' + event_kind, canonical(redact(dict(record, kind=event_kind)))))
        return record

    def claim(self, key, **fields):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT data FROM fixes WHERE id=?', (key,)).fetchone()
            if not row:
                return None
            record = json.loads(row[0])
            if record.get('attempted') or record['phase'] != 'queued':
                return None
            record.update(fields, attempted=True, phase='starting', started_at=time.time(), supervisor_pid=os.getpid())
            db.execute('UPDATE fixes SET data=? WHERE id=?', (canonical(record), key))
            # Commit the notification with the claim: a crash must not lose the
            # start event, and duplicate claimers must not enqueue another one.
            db.execute('INSERT OR IGNORE INTO events(id,data) VALUES (?,?)',
                       (key + ':starting', canonical(redact(dict(record, kind='starting')))))
        return record

    def event(self, key, data):
        with self.connect() as db:
            db.execute('INSERT OR IGNORE INTO events(id,data) VALUES (?,?)', (key, canonical(redact(data))))

    def enqueue(self, task, slot):
        with self.connect() as db:
            db.execute('INSERT OR IGNORE INTO scheduled VALUES (?,?,?,?)', (f'{task}:{slot}', task, 'queued', time.time()))

    def scheduled(self):
        with self.connect(readonly=True) as db:
            return db.execute("SELECT id,task FROM scheduled WHERE state='queued' ORDER BY created,id").fetchall()

    def schedule_state(self, key, state):
        with self.connect() as db:
            db.execute('UPDATE scheduled SET state=? WHERE id=?', (state, key))
