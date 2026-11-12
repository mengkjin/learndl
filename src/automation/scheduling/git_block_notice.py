"""Durable, observation-based Git operation blockage notifications."""
import json
import time
import uuid
from datetime import datetime, timezone

WAIT_SECONDS = 900
MARKERS = ('MERGE_HEAD', 'REBASE_HEAD', 'CHERRY_PICK_HEAD', 'REVERT_HEAD',
           'rebase-merge', 'rebase-apply', 'sequencer', 'index.lock')


def stamp(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


def observe(store, root, markers, *, now=None):
    from src.proj import MACHINE
    now = time.time() if now is None else now
    repository = str(root.resolve())
    with store.connect() as db:
        db.execute('CREATE TABLE IF NOT EXISTS git_operation_notice '
                   '(repository TEXT PRIMARY KEY, first_seen REAL NOT NULL, notice_id TEXT)')
        db.execute('BEGIN IMMEDIATE')
        if not markers:
            db.execute('DELETE FROM git_operation_notice WHERE repository=?', (repository,))
            return
        db.execute('INSERT OR IGNORE INTO git_operation_notice VALUES (?,?,NULL)', (repository, now))
        first, notice_id = db.execute('SELECT first_seen,notice_id FROM git_operation_notice WHERE repository=?',
                                     (repository,)).fetchone()
        if notice_id or now - first < WAIT_SECONDS:
            return
        key = 'operation-' + uuid.uuid4().hex
        body = (f'Host: {MACHINE.name}\nRepository: {repository}\n'
                'Operation: Automatic Git update (origin/master)\n'
                f'First observed blocked: {stamp(first)}\nEvidence captured: {stamp(now)}\n'
                f'Observed wait: {int(now-first)} seconds\n'
                'Reason: Git lock or unfinished operation markers prevent automatic update.\n')
        for item in markers:
            body += f'Marker: {item["path"]}; modified: {stamp(item["mtime"])}; size: {item["size"]} bytes\n'
        # Diagnostic evidence only; proximity never proves ownership of a lock.
        try:
            from .config import runtime_dir
            audit = json.loads((runtime_dir() / 'git-timeout.json').read_text())
            if audit.get('repository') == repository:
                body += 'Last updater timeout (not proof of lock ownership): ' + json.dumps(audit) + '\n'
        except (OSError, ValueError):
            body += 'Updater timeout evidence: unavailable.\n'
        body += ('No locks were removed. File age does not establish that a lock is stale.\n'
                 'Inspect system-level service logs: sudo journalctl -u learndl-git-update.service -n 100 --no-pager\n'
                 'Pause the actual Git timer and verify Git processes/open files before manually moving a stale lock. '
                 'Do not remove .git/index or unfinished merge/rebase markers as a lock repair.\n'
                 'This continuous episode is notified once; delivery retries reuse this snapshot. '
                 'A subsequent observation without these markers resets notification suppression.')
        db.execute('INSERT INTO git_update_mail(id,data) VALUES (?,?)',
                   (key, json.dumps(dict(kind='git_operation', host=MACHINE.name, body=body))))
        db.execute('UPDATE git_operation_notice SET notice_id=? WHERE repository=?', (key, repository))
