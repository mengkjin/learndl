"""RCQuant per-day exclusion and atomic publication; no provider calls here."""
from __future__ import annotations

from typing import Any

__all__ = [
    'sample_size',
    'date_lock',
    'locked_day',
    'atomic_frame',
]
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import errno
import os
import shutil
import sqlite3
from pathlib import Path
import uuid

import portalocker

RAW_IPC = 'zstd'
BAR_IPC = 'uncompressed'
GIB = 1 << 30
MIB = 1 << 20


class SpaceDeferred(Exception):
    """Pause nightly writes until the destination has enough free space."""


def disk_full(exc):
    while exc is not None:
        if (getattr(exc, 'errno', None) == errno.ENOSPC
                or getattr(exc, 'sqlite_errorcode', None) == sqlite3.SQLITE_FULL
                or 'no space left on device' in str(exc).lower()
                or 'os error 28' in str(exc).lower()):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def filesystem_path(path):
    path = Path(path)
    while not path.exists():
        path = path.parent
    return path


def check_space(path, write_bytes, reserve, *, operation):
    free = shutil.disk_usage(filesystem_path(path)).free
    if free < reserve + write_bytes:
        raise SpaceDeferred(f'{operation} on {path}: free {free / GIB:.2f} GiB, '
                            f'need {(reserve + write_bytes) / GIB:.2f} GiB '
                            f'({reserve / GIB:g} GiB reserve)')


def frame_bytes(frame):
    # IPC bodies contain the column buffers, plus schema, message metadata and padding.
    return int(frame.estimated_size() if hasattr(frame, 'estimated_size')
               else frame.memory_usage(index=True, deep=True).sum()) + MIB


sample_size = ContextVar('rcquant_sample_size', default=-1)


@contextmanager
def date_lock(root, data_type, date):
    directory = Path(root) / '.locks'
    directory.mkdir(parents=True, exist_ok=True)
    # Different dates can progress independently; one date cannot be fetched twice.
    with (directory / f'{data_type}-{int(date)}.lock').open('a+') as stream:
        portalocker.lock(stream, portalocker.LOCK_EX)
        yield


def locked_day(func):
    @wraps(func)
    def wrapped(self, date, data_type, first_n=-1, **kwargs):
        from .bar_min import RQ_PATH
        with date_lock(RQ_PATH, data_type, date):
            token = sample_size.set(first_n)
            try:
                return func(self, date, data_type, first_n, **kwargs)
            finally:
                sample_size.reset(token)
    return wrapped


def atomic_frame(frame, path, *, prefix=None, indent=1, vb_level: Any = 1,
                 ipc_compression=None, empty_ok=False, **kwargs):
    from src.proj import Logger, Save
    if frame is None or (not empty_ok and len(frame) == 0):
        raise OSError(f'RCQuant publication has no rows: {path}')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the final extension so the existing serializer chooses the same format.
    temp = path.with_name(f'.{path.stem}-{uuid.uuid4().hex}.tmp{path.suffix}')
    try:
        if ipc_compression is None:
            result = Save.df(frame, temp, async_save=False, vb_level='never', empty_ok=empty_ok, **kwargs)
        else:
            import polars as pl
            if not isinstance(frame, pl.DataFrame):
                frame = pl.from_pandas(frame, nan_to_null=False)
            frame.rechunk().write_ipc(temp, compression=ipc_compression)
            result = True
        if result is False or not temp.exists():
            raise OSError(f'RCQuant publication failed: {path}')
        with temp.open('rb') as stream:
            os.fsync(stream.fileno())
        status = 'Overwritten ' if path.exists() else 'File Created'
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    if vb_level == 'never':
        return True
    Logger.footnote(f'{prefix or ""} {status}: {path}', indent=indent, vb_level=vb_level)
    return True
