"""Filesystem helpers for atomic writes and path validation."""
from __future__ import annotations
import json
import os
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

__all__ = ['atomic_bytes', 'atomic_json', 'fsync_dir', 'real_tree']


@contextmanager
def _atomic_path(path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + uuid4().hex)
    try:
        yield tmp
        os.replace(tmp, path)
        fsync_dir(path.parent)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_bytes(path: str | Path, value: bytes) -> None:
    """Write bytes, then atomically publish the complete file."""
    with _atomic_path(path) as tmp, tmp.open('xb') as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def atomic_json(path: str | Path, value) -> None:
    """Stream JSON to a temporary file and publish it as UTF-8."""
    with _atomic_path(path) as tmp, tmp.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())


def fsync_dir(path):
    if os.name == 'nt':
        # Windows cannot open directories through the CRT file descriptor API.
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def real_tree(path):
    path = Path(path).absolute()
    for p in (path, *path.parents):
        if p.is_symlink():
            raise ValueError(f'Symbolic link refused: {p}')
    return path
