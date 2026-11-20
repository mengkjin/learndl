"""Lightweight filesystem helpers for local state and artifact exchange."""
from __future__ import annotations
import json
import os
from pathlib import Path
from uuid import uuid4

__all__ = ['atomic_json', 'fsync_dir', 'real_tree']


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + uuid4().hex)
    try:
        with tmp.open('x') as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        fsync_dir(path.parent)
    finally:
        tmp.unlink(missing_ok=True)


def fsync_dir(path):
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
