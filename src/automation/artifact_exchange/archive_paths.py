"""Update durable references when received artifact directories move."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from src.shared.file_ops import atomic_json, real_tree

__all__ = ['remap', 'update_state']


def read(path):
    return json.loads(real_tree(path).read_text())


def remap(path, mapping):
    """Translate a stored path with its most specific directory mapping."""
    for parent in (path, *path.parents):
        if parent in mapping:
            return mapping[parent] / path.relative_to(parent)
    return path


def update_state(config, monitor, mapping):
    """Retarget receipts, trash, running dashboards and archived file IDs after a move."""
    def update_field(path, keys):
        record = read(path)
        updated = dict(record)
        for key in keys:
            if record.get(key):
                updated[key] = str(remap(Path(record[key]), mapping))
        if updated != record:
            atomic_json(path, updated)

    for receipt in (config.state / 'received').glob('*.json'):
        update_field(receipt, ('target',))
    for meta in monitor.glob('*/trash/*/item.json'):
        update_field(meta, ('original',))
    for record in (monitor / 'experiments').glob('[0-9]*.json'):
        update_field(record, ('archive_source', 'source'))
    marks = list(monitor.glob('*/archived.json'))
    if any(read(path) for path in marks):
        ids = {}
        for src, dest in mapping.items():
            if dest.exists():
                for actual in (dest, *dest.rglob('*')):
                    old = src / actual.relative_to(dest)
                    ids[sha256(str(old).encode()).hexdigest()] = sha256(str(actual).encode()).hexdigest()
        for path in marks:
            old = read(path)
            new = sorted({ids.get(key, key) for key in old})
            if old != new:
                atomic_json(path, new)
