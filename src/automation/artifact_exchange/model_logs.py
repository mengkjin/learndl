"""Compatibility for the pre-0.7.7 receiver’s Markdown fragments."""
from __future__ import annotations

import hashlib
import json
import os

from src.model_provenance import LOG_PHASES
from .core import atomic_json, component, fsync_dir, real_tree

__all__ = ['log_root', 'append_received']


def log_root(config, host, model):
    return real_tree(config.results / 'external/models' / component(host) / component(model) / 'run_logs')


def append_received(config, target, manifest):
    """Recover an interrupted append using its durable offset before writing again."""
    from contextlib import closing
    import sqlite3

    if manifest.get('category') != 'model_training':
        return
    folder = log_root(config, manifest['host'], manifest['experiment'])
    for phase in LOG_PHASES:
        name = f'run-log/{phase}.md'
        if not any(item['path'] == name for item in manifest['files']):
            continue
        fragment = real_tree(target / name)
        content = fragment.read_bytes()
        folder.mkdir(parents=True, exist_ok=True)
        atomic_json(folder / 'model.json', {'host': manifest['host'], 'model': manifest['experiment']})
        path = real_tree(folder / f'{phase}.md')
        pending = folder / f'.{phase}.pending.json'
        with closing(sqlite3.connect(folder / '.runs.sqlite3')) as db, db:
            db.execute('CREATE TABLE IF NOT EXISTS runs (phase TEXT, id TEXT, sha TEXT, PRIMARY KEY (phase, id))')
        key = (phase, manifest['run_id'])
        sha = hashlib.sha256(content).hexdigest()
        with closing(sqlite3.connect(folder / '.runs.sqlite3')) as db:
            previous = db.execute('SELECT sha FROM runs WHERE phase=? AND id=?', key).fetchone()
            if previous:
                if previous[0] != sha:
                    raise ValueError('Model run ID reused with different log content')
                if pending.exists() and json.loads(pending.read_text())['id'] == key[1]:
                    pending.unlink()
                    fsync_dir(folder)
                continue
            if pending.exists():
                journal = json.loads(pending.read_text())
                if (journal['id'], journal['sha']) != (key[1], sha):
                    raise ValueError('Prior model log append must finish first')
            else:
                journal = {'id': key[1], 'sha': sha, 'offset': path.stat().st_size if path.exists() else 0}
                atomic_json(pending, journal)
            offset = journal['offset']
            with path.open('a+b') as stream:
                if stream.seek(0, os.SEEK_END) < offset:
                    raise ValueError('Model log changed during an unfinished append')
                stream.truncate(offset)
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            fsync_dir(folder)
            with db:
                db.execute('INSERT INTO runs VALUES (?, ?, ?)', (*key, sha))
            pending.unlink()
            fsync_dir(folder)
